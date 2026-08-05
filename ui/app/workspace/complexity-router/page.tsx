import FullPageLoader from "@/components/fullPageLoader";
import {
	AlertDialog,
	AlertDialogAction,
	AlertDialogCancel,
	AlertDialogContent,
	AlertDialogDescription,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogTitle,
} from "@/components/ui/alertDialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ModelMultiselect } from "@/components/ui/modelMultiselect";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scrollArea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { TagInput } from "@/components/ui/tagInput";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ProviderIconType, RenderProviderIcon } from "@/lib/constants/icons";
import { EmbeddingSupportedProviders, getProviderLabel } from "@/lib/constants/logs";
import { getErrorMessage, useGetCoreConfigQuery, useGetProvidersQuery } from "@/lib/store";
import {
	useGetComplexityAnalyzerConfigQuery,
	useGetComplexitySemanticStatusQuery,
	useProbeComplexityEmbeddingDimensionMutation,
	useResetComplexityAnalyzerConfigMutation,
	useUpdateComplexityAnalyzerConfigMutation,
} from "@/lib/store/apis/governanceApi";
import {
	AnalyzerConfig,
	DEFAULT_SEMANTIC_CONFIG,
	DEFAULT_TIER_BOUNDARIES,
	KeywordListKey,
	MAX_SEMANTIC_MESSAGE_HISTORY,
	MAX_SEMANTIC_PHRASE_CHARACTERS,
	MIN_SEMANTIC_MESSAGE_HISTORY,
	parseSemanticTimeoutMs,
	SEMANTIC_STATUS_LABELS,
	SEMANTIC_VECTOR_STORE_OPTIONS,
	SemanticStatusInfo,
	TIER_PHRASE_LIST_DEFINITIONS,
	TierBoundaries,
} from "@/lib/types/complexityRouter";
import { ModelProvider, ModelProviderName } from "@/lib/types/config";
import { cn } from "@/lib/utils";
import { RbacOperation, RbacResource, useRbac } from "@enterprise/lib";
import { zodResolver } from "@hookform/resolvers/zod";
import { CircleAlert, CircleCheck, ExternalLink, Info, LoaderCircle, RotateCcw, Save, TriangleAlert } from "lucide-react";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { Controller, useForm } from "react-hook-form";
import { toast } from "sonner";
import { z } from "zod";

// Embedding-capable providers gate this page, matching the local cache screen's
// rule: built-ins are listed in EmbeddingSupportedProviders, custom providers
// declare support through allowed_requests.embedding. A custom provider with no
// allowed_requests block at all is unrestricted, which is how the Go side reads
// a nil AllowedRequests.
const supportsEmbedding = (provider: ModelProvider): boolean => {
	if (provider.custom_provider_config) {
		const allowed = provider.custom_provider_config.allowed_requests;
		return !allowed || allowed.embedding === true;
	}
	return (EmbeddingSupportedProviders as readonly string[]).includes(provider.name);
};

// The three tier lists sit side by side, so they collapse to a fixed height
// rather than to a fixed number of phrases: phrases wrap to different numbers of
// lines, and equal counts would leave the columns visibly uneven.
const PHRASE_LIST_COLLAPSED_HEIGHT = 260;

const semanticSchema = z.object({
	provider: z.string(),
	embedding_model: z.string(),
	// Measured by the dimension probe, never typed, so the message describes a
	// failed or missing detection rather than bad input.
	dimension: z.number().int().min(0),
	// The control edits milliseconds but the value stays a Go duration, so a
	// non-positive or malformed entry is caught here rather than snapped back to
	// the default while the operator is still typing.
	timeout: z
		.string()
		.min(1, "Enter an embedding timeout")
		.refine(
			(value) => /^[0-9]*\.?[0-9]+(ns|us|µs|ms|s|m|h)$/.test(value.trim()) && Number.parseFloat(value) > 0,
			"Enter a timeout greater than 0",
		)
		.optional(),
	fallback: z.enum(["lexical", "none"]).optional(),
	min_similarity: z.number({ error: "Enter a number between 0 and 1" }).min(0, "Must be 0 or greater").lt(1, "Must be less than 1"),
	message_history_count: z
		.number({ error: `Enter a number between ${MIN_SEMANTIC_MESSAGE_HISTORY} and ${MAX_SEMANTIC_MESSAGE_HISTORY}` })
		.int("Must be a whole number")
		.min(MIN_SEMANTIC_MESSAGE_HISTORY, `Must be at least ${MIN_SEMANTIC_MESSAGE_HISTORY}`)
		.max(MAX_SEMANTIC_MESSAGE_HISTORY, `Must be at most ${MAX_SEMANTIC_MESSAGE_HISTORY}`),
	count_toward_budgets: z.boolean().optional(),
	vector_store: z.enum(["auto", "embedded"]).optional(),
});

const analyzerConfigSchema = z
	.object({
		// Not editable on this page. The lexical scorer still reads them, and the
		// API rejects a config without them, so they are carried through untouched.
		tier_boundaries: z.object({
			simple_medium: z.number(),
			medium_complex: z.number(),
		}),
		keywords: z.object({
			simple_keywords: z.array(z.string()).min(1, "Simple phrases cannot be empty"),
			medium_keywords: z.array(z.string()).min(1, "Medium phrases cannot be empty"),
			complex_keywords: z.array(z.string()).min(1, "Complex phrases cannot be empty"),
		}),
		semantic: semanticSchema,
	})
	.superRefine((data, ctx) => {
		// A blank provider and model means the classifier simply is not configured
		// yet, which is a legal state: phrase edits still save. Half-filled is not,
		// because it cannot be turned into a working classifier.
		const hasProvider = data.semantic.provider.trim() !== "";
		const hasModel = data.semantic.embedding_model.trim() !== "";
		if (hasProvider || hasModel) {
			if (!hasProvider) {
				ctx.addIssue({ code: "custom", message: "Select an embedding provider", path: ["semantic", "provider"] });
			}
			if (!hasModel) {
				ctx.addIssue({ code: "custom", message: "Select an embedding model", path: ["semantic", "embedding_model"] });
			}
			if (hasProvider && hasModel && data.semantic.dimension < 2) {
				ctx.addIssue({
					code: "custom",
					message: "The model's embedding dimension could not be detected. Reselect the model to retry.",
					path: ["semantic", "embedding_model"],
				});
			}
		}

		// Mirrors validateComplexitySemanticPhrases so invalid input fails in the
		// form instead of as an opaque 400.
		const lists: Array<{ key: KeywordListKey; label: string }> = [
			{ key: "simple_keywords", label: "Simple" },
			{ key: "medium_keywords", label: "Medium" },
			{ key: "complex_keywords", label: "Complex" },
		];

		const seen = new Map<string, string>();
		for (const { key, label } of lists) {
			for (const phrase of data.keywords[key]) {
				if (phrase.length > MAX_SEMANTIC_PHRASE_CHARACTERS) {
					ctx.addIssue({
						code: "custom",
						message: `A ${label} phrase exceeds the ${MAX_SEMANTIC_PHRASE_CHARACTERS}-character limit.`,
						path: ["keywords", key],
					});
					break;
				}
				const normalized = phrase.trim().toLowerCase();
				const firstTier = seen.get(normalized);
				if (firstTier && firstTier !== label) {
					ctx.addIssue({
						code: "custom",
						message: `"${phrase}" is also in the ${firstTier} list. Each phrase must belong to exactly one tier.`,
						path: ["keywords", key],
					});
				} else if (!firstTier) {
					seen.set(normalized, label);
				}
			}
		}
	});

// The form is stricter than the wire type: the API omits semantic fields left at
// their zero value (Go `omitempty`), but every control here is controlled and
// needs a concrete value, so the schema's inferred type is the source of truth.
type AnalyzerFormValues = z.infer<typeof analyzerConfigSchema>;
type SemanticFormValues = AnalyzerFormValues["semantic"];

const DEFAULT_SEMANTIC_FORM_VALUES: SemanticFormValues = {
	...DEFAULT_SEMANTIC_CONFIG,
	min_similarity: DEFAULT_SEMANTIC_CONFIG.min_similarity ?? 0,
	message_history_count: DEFAULT_SEMANTIC_CONFIG.message_history_count ?? MIN_SEMANTIC_MESSAGE_HISTORY,
	vector_store: "embedded",
};

const DEFAULT_FORM_VALUES: AnalyzerFormValues = {
	tier_boundaries: { ...DEFAULT_TIER_BOUNDARIES },
	keywords: {
		simple_keywords: [],
		medium_keywords: [],
		complex_keywords: [],
	},
	semantic: DEFAULT_SEMANTIC_FORM_VALUES,
};

// Boundaries have no control on this page, so an out-of-range persisted value
// could never be corrected and would block every save. Fall back to the defaults
// instead, which is what the lexical scorer would use anyway.
function usableBoundaries(boundaries: TierBoundaries | undefined): TierBoundaries {
	const simpleMedium = boundaries?.simple_medium;
	const mediumComplex = boundaries?.medium_complex;
	const ordered =
		typeof simpleMedium === "number" &&
		typeof mediumComplex === "number" &&
		Number.isFinite(simpleMedium) &&
		Number.isFinite(mediumComplex) &&
		0 < simpleMedium &&
		simpleMedium < mediumComplex &&
		mediumComplex < 1;
	return ordered ? { simple_medium: simpleMedium, medium_complex: mediumComplex } : { ...DEFAULT_TIER_BOUNDARIES };
}

// Fills in the fields the API omitted so the semantic controls stay controlled.
function toFormValues(config: AnalyzerConfig): AnalyzerFormValues {
	const saved = config.semantic;
	return {
		tier_boundaries: usableBoundaries(config.tier_boundaries),
		keywords: config.keywords,
		semantic: saved
			? {
					...DEFAULT_SEMANTIC_FORM_VALUES,
					...saved,
					min_similarity: saved.min_similarity ?? 0,
					message_history_count: saved.message_history_count ?? MIN_SEMANTIC_MESSAGE_HISTORY,
					// The lexical classifier is no longer offered, so a config still
					// carrying fallback "lexical" is normalized on the next save.
					fallback: "none",
					// "external" is still a valid wire value but is no longer offered:
					// it turns a missing vector store into a startup error. It reads,
					// and re-saves, as the falling-back "auto".
					vector_store: saved.vector_store === "embedded" ? "embedded" : "auto",
				}
			: DEFAULT_SEMANTIC_FORM_VALUES,
	};
}

function testIdPart(value: string) {
	return value.replace(/_/g, "-");
}

// The timeout control edits milliseconds while the form value stays a Go
// duration. A value this control wrote round-trips digit for digit, including a
// "0" the operator is midway through typing, which the schema rejects rather
// than the field silently rewriting. Anything else — a saved "1s", a blank —
// falls back to the parsed reading.
function semanticTimeoutFieldValue(timeout: string | undefined): string | number {
	if (timeout === "") return "";
	const millis = timeout?.trim().match(/^([0-9]*\.?[0-9]+)ms$/);
	return millis ? millis[1] : parseSemanticTimeoutMs(timeout);
}

// A tap is neither a hover nor a keyboard focus, so a Radix tooltip never opens
// on a touch device. Coarse pointers get the same copy from a popover instead.
function useCoarsePointer() {
	const [isCoarse, setIsCoarse] = useState(false);
	useEffect(() => {
		const query = window.matchMedia("(pointer: coarse)");
		const sync = () => setIsCoarse(query.matches);
		sync();
		query.addEventListener("change", sync);
		return () => query.removeEventListener("change", sync);
	}, []);
	return isCoarse;
}

// InfoTip carries the explanation that used to sit under a field, so the page
// reads as a form rather than as documentation.
function InfoTip({ label, children }: { label: string; children: ReactNode }) {
	const isCoarsePointer = useCoarsePointer();
	const trigger = (
		<button type="button" aria-label={label} className="text-muted-foreground/70 hover:text-foreground transition-colors">
			<Info className="size-3.5" />
		</button>
	);

	if (isCoarsePointer) {
		return (
			<Popover>
				<PopoverTrigger asChild>{trigger}</PopoverTrigger>
				<PopoverContent className="w-auto max-w-xs p-3 text-xs leading-relaxed">{children}</PopoverContent>
			</Popover>
		);
	}

	return (
		<Tooltip>
			<TooltipTrigger asChild>{trigger}</TooltipTrigger>
			<TooltipContent className="max-w-xs leading-relaxed">{children}</TooltipContent>
		</Tooltip>
	);
}

function FieldLabel({ htmlFor, children, tooltip }: { htmlFor?: string; children: ReactNode; tooltip?: ReactNode }) {
	return (
		<div className="flex items-center gap-1.5">
			<Label htmlFor={htmlFor}>{children}</Label>
			{tooltip && <InfoTip label={`About ${typeof children === "string" ? children : "this field"}`}>{tooltip}</InfoTip>}
		</div>
	);
}

function Callout({ tone = "info", children, testId }: { tone?: "info" | "warning"; children: ReactNode; testId?: string }) {
	const Icon = tone === "warning" ? TriangleAlert : Info;
	return (
		<div
			data-testid={testId}
			className={cn(
				"flex items-start gap-2 rounded-sm border px-3 py-2 text-xs leading-relaxed",
				tone === "warning"
					? "border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200"
					: "bg-muted/40 text-muted-foreground",
			)}
		>
			<Icon className="mt-0.5 size-3.5 shrink-0" />
			<div>{children}</div>
		</div>
	);
}

function SectionHeading({ title, description, aside }: { title: string; description: string; aside?: ReactNode }) {
	return (
		<div className="flex flex-wrap items-start justify-between gap-2">
			<div className="space-y-1">
				<h2 className="text-sm font-semibold">{title}</h2>
				<p className="text-muted-foreground max-w-2xl text-xs leading-relaxed">{description}</p>
			</div>
			{aside}
		</div>
	);
}

// SemanticStatusPanel surfaces warmup readiness, which is otherwise only visible
// in server logs. Without it a failed warmup looks identical to a working
// deployment, because complexity routing simply stops matching.
function SemanticStatusPanel({
	status,
	isLoading,
	isNotConfigured,
	isNotSaved,
	hasUnsavedChanges,
}: {
	status: SemanticStatusInfo | undefined;
	isLoading: boolean;
	isNotConfigured: boolean;
	isNotSaved: boolean;
	hasUnsavedChanges: boolean;
}) {
	if (isNotConfigured || isNotSaved) {
		return (
			<div className="bg-card space-y-3 rounded-sm border p-4" data-testid="complexity-router-semantic-status">
				<div className="flex items-center justify-end">
					<Badge
						className="border-0 bg-blue-100 py-0.5 text-[10px] text-blue-800 uppercase dark:bg-blue-900 dark:text-blue-300"
						data-testid="complexity-router-semantic-status-badge"
					>
						{isNotConfigured ? "Not configured" : "Not saved"}
					</Badge>
				</div>
				<p className="text-muted-foreground text-xs">
					{isNotConfigured
						? "Select an embedding provider and model below, then save to embed the reference phrases and activate classification."
						: "Save this configuration to embed the reference phrases and activate classification."}
				</p>
			</div>
		);
	}

	if (isLoading && !status) {
		return (
			<div className="bg-card text-muted-foreground flex items-center gap-2 rounded-sm border p-4 text-xs">
				<LoaderCircle className="size-3.5 animate-spin" />
				Checking classifier status…
			</div>
		);
	}
	if (!status) return null;

	const tone = {
		ready: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300",
		warming: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300",
		failed: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
		disabled: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300",
	}[status.state];

	const percent = status.total > 0 ? Math.round((status.loaded / status.total) * 100) : 0;

	return (
		<div className="bg-card space-y-3 rounded-sm border p-4" data-testid="complexity-router-semantic-status">
			<div className="flex items-center justify-between">
				<div className="flex items-center gap-2">
					{status.state === "ready" ? (
						<CircleCheck className="size-3.5 text-green-600" />
					) : status.state === "failed" ? (
						<CircleAlert className="text-destructive size-3.5" />
					) : status.state === "warming" ? (
						<LoaderCircle className="size-3.5 animate-spin text-amber-600" />
					) : (
						<span />
					)}
					<span className="text-muted-foreground text-xs">
						{status.state === "warming"
							? "Embedding reference phrases"
							: status.state === "ready"
								? `${status.total} reference phrase${status.total === 1 ? "" : "s"} embedded and serving`
								: status.state === "failed"
									? "Warmup failed"
									: "Classification is off"}
					</span>
				</div>
				<Badge className={cn("border-0 py-0.5 text-[10px] uppercase", tone)} data-testid="complexity-router-semantic-status-badge">
					{SEMANTIC_STATUS_LABELS[status.state]}
				</Badge>
			</div>

			{status.state === "warming" && (
				<div className="space-y-1.5">
					<Progress value={percent} className="h-1.5" />
					<p className="text-muted-foreground font-mono text-[11px] tabular-nums">
						{status.loaded}/{status.total}
					</p>
				</div>
			)}

			{status.serving_previous && (
				<p className="text-xs text-amber-700 dark:text-amber-400">
					The previous reference phrases are still serving requests while this generation prepares. Routing is unaffected.
				</p>
			)}

			{hasUnsavedChanges && (
				<p className="text-xs text-amber-700 dark:text-amber-400">
					The saved classifier is still serving. Save to prepare and activate these phrase or model changes.
				</p>
			)}

			{status.error && (
				<p className="text-destructive text-xs" data-testid="complexity-router-semantic-status-error">
					{status.error}
				</p>
			)}

			{status.state === "failed" && (
				<p className="text-destructive text-xs">
					Until warmup succeeds, complexity tier based routing is skipped, so rules referencing{" "}
					<code className="font-mono">complexity_tier</code> do not match.
				</p>
			)}
		</div>
	);
}

export default function ComplexityRouterPage() {
	const canUpdate = useRbac(RbacResource.RoutingRules, RbacOperation.Update);
	const { data, isLoading, isFetching, error, refetch } = useGetComplexityAnalyzerConfigQuery();
	const [updateConfig, { isLoading: isSaving }] = useUpdateComplexityAnalyzerConfigMutation();
	const [resetConfig, { isLoading: isResetting }] = useResetComplexityAnalyzerConfigMutation();

	const [submitError, setSubmitError] = useState<string | null>(null);
	const [restoreDialogOpen, setRestoreDialogOpen] = useState(false);
	// Monotonic id for dimension probes, so a late reply from a superseded
	// provider/model pair cannot clobber the current one.
	const dimensionProbeSeqRef = useRef(0);

	const { data: providersData, isLoading: providersLoading } = useGetProvidersQuery();
	const embeddingProviders = useMemo(() => (providersData || []).filter(supportsEmbedding), [providersData]);

	const { data: coreConfig } = useGetCoreConfigQuery({ fromDB: true });
	const isVectorStoreConnected = coreConfig?.is_cache_connected ?? false;

	const [probeDimension, { isLoading: isProbingDimension }] = useProbeComplexityEmbeddingDimensionMutation();
	const [dimensionProbeError, setDimensionProbeError] = useState<string | null>(null);

	// Poll only while warmup is in flight; a ready or failed classifier is steady
	// state until the next save, which refetches through the cache tag.
	const [statusPollInterval, setStatusPollInterval] = useState(0);
	const { data: semanticStatus, isLoading: statusLoading } = useGetComplexitySemanticStatusQuery(undefined, {
		skip: !data?.semantic,
		pollingInterval: statusPollInterval,
	});
	useEffect(() => {
		setStatusPollInterval(semanticStatus?.state === "warming" ? 2000 : 0);
	}, [semanticStatus?.state]);

	const {
		register,
		handleSubmit,
		reset,
		control,
		watch,
		setValue,
		formState: { errors, isDirty, isSubmitted },
	} = useForm<AnalyzerFormValues>({
		resolver: zodResolver(analyzerConfigSchema),
		defaultValues: DEFAULT_FORM_VALUES,
		mode: "onSubmit",
		reValidateMode: "onChange",
	});

	const liveSemantic = watch("semantic");
	const liveKeywords = watch("keywords");

	const isClassifierConfigured = Boolean(liveSemantic?.provider && liveSemantic?.embedding_model);

	const totalPhrases = useMemo(
		() =>
			(liveKeywords?.simple_keywords?.length ?? 0) +
			(liveKeywords?.medium_keywords?.length ?? 0) +
			(liveKeywords?.complex_keywords?.length ?? 0),
		[liveKeywords],
	);

	// Saving re-runs warmup. It is a no-op when only the threshold or timeout
	// changed (the phrase fingerprint is unchanged), but re-embeds every phrase
	// when the provider, model, dimension, or a list changed.
	const willReembed = useMemo(() => {
		if (!data || !isClassifierConfigured) return false;
		const saved = data.semantic;
		if (!saved) return true;
		return (
			saved.provider !== liveSemantic?.provider ||
			saved.embedding_model !== liveSemantic?.embedding_model ||
			saved.dimension !== liveSemantic?.dimension ||
			JSON.stringify(data.keywords) !== JSON.stringify(liveKeywords)
		);
	}, [data, isClassifierConfigured, liveSemantic, liveKeywords]);

	useEffect(() => {
		if (!data || isDirty) return;
		reset(toFormValues(data));
		setSubmitError(null);
	}, [data, isDirty, reset]);

	// Probing costs a real embedding call, so it runs only when the operator picks
	// a provider or model, never on page load, where the saved dimension is
	// already correct for the saved model.
	const runDimensionProbe = async (provider: string, model: string) => {
		if (!provider || !model) return;
		// Switching models twice in a row leaves two probes in flight, and the
		// first can land last. Only the newest request may write the dimension.
		const seq = ++dimensionProbeSeqRef.current;
		setDimensionProbeError(null);
		try {
			const { dimension } = await probeDimension({ provider, embedding_model: model }).unwrap();
			if (seq !== dimensionProbeSeqRef.current) return;
			setValue("semantic.dimension", dimension, { shouldDirty: true, shouldValidate: true });
		} catch (probeError) {
			if (seq !== dimensionProbeSeqRef.current) return;
			// Leave the dimension unset rather than stale: saving a width that
			// belongs to a previously selected model is the exact failure this probe
			// exists to prevent.
			setValue("semantic.dimension", 0, { shouldDirty: true });
			setDimensionProbeError(getErrorMessage(probeError));
		}
	};

	const handleDiscard = () => {
		if (data) reset(toFormValues(data));
		setDimensionProbeError(null);
		setSubmitError(null);
	};

	const handleRestoreDefaults = () => {
		if (!canUpdate) return;
		setSubmitError(null);
		resetConfig()
			.unwrap()
			.then((defaults) => {
				reset(toFormValues(defaults));
				setDimensionProbeError(null);
				toast.success("Reset to defaults", { position: "top-right" });
			})
			.catch((err) => {
				setSubmitError(getErrorMessage(err));
			});
	};

	const onValid = (values: AnalyzerFormValues) => {
		if (!canUpdate) return;
		setSubmitError(null);
		// The endpoint replaces the whole record and rejects a semantic block
		// without a provider and model, so an unconfigured classifier omits it
		// entirely and saves the phrase lists alone.
		const payload: AnalyzerConfig = {
			tier_boundaries: values.tier_boundaries,
			keywords: values.keywords,
			...(values.semantic.provider && values.semantic.embedding_model ? { semantic: values.semantic } : {}),
		};
		updateConfig(payload)
			.unwrap()
			.then((res) => {
				reset(toFormValues(res));
				toast.success("Configuration saved", { position: "top-right" });
			})
			.catch((err) => {
				setSubmitError(getErrorMessage(err));
			});
	};

	if (isLoading && !data) {
		return <FullPageLoader />;
	}

	if (error && !data) {
		return (
			<div className="mx-auto w-full max-w-7xl space-y-4 px-14 pt-8">
				<p className="text-destructive font-mono text-sm">{getErrorMessage(error)}</p>
				<Button data-testid="complexity-router-fetch-retry-button" type="button" variant="outline" size="sm" onClick={() => refetch()}>
					Retry
				</Button>
			</div>
		);
	}

	if (!data) {
		return (
			<div className="mx-auto w-full max-w-7xl space-y-4 px-14 pt-8">
				<p className="text-muted-foreground font-mono text-sm">No complexity router configuration is available.</p>
				<Button data-testid="complexity-router-fetch-retry-button" type="button" variant="outline" size="sm" onClick={() => refetch()}>
					Retry
				</Button>
			</div>
		);
	}

	const keywordErrors = errors.keywords;
	const semanticErrors = errors.semantic;
	const hasErrors = Boolean(errors.tier_boundaries || keywordErrors || semanticErrors);
	// An in-flight providers query yields an empty list too, which is not the same
	// as having none: gating on it would fire on every page load and blame the
	// operator for a provider they already configured.
	const noEmbeddingProviders = !providersLoading && embeddingProviders.length === 0;

	return (
		<>
			<form className="no-padding-parent own-scroll-parent flex h-full min-h-0 w-full flex-col" onSubmit={handleSubmit(onValid)} noValidate>
				{/* The footer is a sibling of the scroll area rather than a sticky child
				    of it. Radix wraps scrolled content in a display:table element, and
				    position:sticky is unreliable inside table boxes: it parked the footer
				    partway up the scrollport and left dead space beneath it. */}
				<ScrollArea className="min-h-0 flex-1 px-14 pt-4">
					<div className="mx-auto w-full max-w-7xl space-y-8 pb-8">
						{/* ── Page header ── */}
						<div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
							<div className="space-y-1.5">
								<h1 className="text-2xl font-semibold tracking-tight">Complexity Router</h1>
								<p className="text-muted-foreground max-w-2xl text-sm leading-relaxed">
									Each request is embedded and takes the tier of the nearest reference phrase, filling the{" "}
									<code className="bg-muted rounded-sm px-1 py-0.5 font-mono text-xs">complexity_tier</code> field that routing rules
									target.
								</p>
							</div>
							<Button asChild variant="outline" size="sm" className="w-fit shrink-0" data-testid="complexity-router-docs-link">
								<a href={"https://docs.getbifrost.ai/features/governance/complexity-router"} target="_blank" rel="noopener noreferrer">
									<ExternalLink className="size-3.5" />
									Docs
								</a>
							</Button>
						</div>

						{noEmbeddingProviders && (
							<Callout tone="warning" testId="complexity-router-no-embedding-providers">
								No embedding-capable provider is configured. Phrase edits still save, but classification stays off until you add one.
							</Callout>
						)}

						{/* ── Phrase to Tier Mapping ── */}
						<div className="space-y-3">
							<SectionHeading
								title="Phrase to Tier Mapping"
								description="Reference/example phrases that define each tier. A request takes the tier of its nearest phrase."
								aside={
									<span className="text-muted-foreground font-mono text-[11px] tabular-nums" data-testid="complexity-router-phrase-total">
										{totalPhrases} phrases
									</span>
								}
							/>

							<Callout testId="complexity-router-phrase-defaults-callout">These are defaults; tune them to your use case.</Callout>

							{/* Root-level phrase issues such as cross-tier duplicates have no single
					    field to attach to, so they render above the lists. */}
							{keywordErrors?.message && (
								<p className="text-destructive text-xs" data-testid="complexity-router-keywords-error">
									{keywordErrors.message}
								</p>
							)}

							{/* One column per tier, side by side: the three lists are read against
					    each other, and equal-width columns keep a phrase's tier obvious
					    from its position. */}
							<div className="grid gap-3 md:grid-cols-3">
								{TIER_PHRASE_LIST_DEFINITIONS.map(({ key, label, description }) => {
									const fieldError = keywordErrors?.[key as KeywordListKey];
									const errorId = `keywords-${key}-error`;
									return (
										<div key={key} className="bg-card relative overflow-hidden rounded-sm border">
											<Controller
												control={control}
												name={`keywords.${key}` as const}
												rules={{ validate: (value) => (value.length > 0 ? true : `${label} phrases cannot be empty`) }}
												render={({ field }) => (
													<div className="space-y-2 p-4 pl-5">
														<div className="flex items-center justify-between">
															<span className="text-xs font-medium">{label}</span>
															<span className="text-muted-foreground font-mono text-[11px] tabular-nums">
																{field.value.length} {field.value.length === 1 ? "phrase" : "phrases"}
															</span>
														</div>
														<p className="text-muted-foreground text-xs leading-relaxed">{description}</p>
														<TagInput
															data-testid={`complexity-router-keywords-${testIdPart(key)}-input`}
															value={field.value}
															onValueChange={field.onChange}
															collapsedMaxHeight={PHRASE_LIST_COLLAPSED_HEIGHT}
															expandButtonTestId={`complexity-router-keywords-${testIdPart(key)}-expand-button`}
															placeholder="Type a reference/example phrase and press Enter"
															aria-invalid={fieldError ? true : undefined}
															aria-describedby={fieldError ? errorId : undefined}
															className={cn(fieldError && "border-destructive")}
														/>
														{fieldError && (
															<p id={errorId} className="text-destructive text-xs">
																{fieldError.message}
															</p>
														)}
													</div>
												)}
											/>
										</div>
									);
								})}
							</div>
						</div>

						{/* ── Classifier Status ── */}
						<div className="space-y-3">
							<SectionHeading title="Classifier Status" description="Whether the saved reference phrases are embedded and serving." />

							<SemanticStatusPanel
								status={semanticStatus}
								isLoading={statusLoading}
								isNotConfigured={!isClassifierConfigured}
								isNotSaved={isClassifierConfigured && !data.semantic}
								hasUnsavedChanges={willReembed}
							/>

							{willReembed && (
								<Callout tone="warning" testId="complexity-router-reembed-warning">
									Saving will embed all {totalPhrases} reference phrases through the selected provider. This uses embedding tokens and may
									take a short time. Changes only to the threshold, timeout, or storage reuse the current embeddings.
								</Callout>
							)}
						</div>

						{/* ── Embedding Configuration ── */}
						<div className="space-y-3">
							<SectionHeading
								title="Embedding Configuration"
								description="The model that embeds requests and reference phrases. API keys are inherited from the provider's main configuration."
							/>

							{/* Everything below the provider and model is part of the semantic
					    block, which is only persisted once both are set. Without this
					    hint the remaining fields look editable but silently reset. */}
							{!providersLoading && !noEmbeddingProviders && !isClassifierConfigured && (
								<Callout testId="complexity-router-classifier-required-callout">
									Pick an embedding provider and model to configure the rest. Until then the classifier is off and only the phrase lists are
									saved.
								</Callout>
							)}

							<div className="bg-card space-y-5 rounded-sm border p-5">
								{providersLoading ? (
									<div className="flex items-center justify-center py-4">
										<LoaderCircle className="text-muted-foreground size-4 animate-spin" />
									</div>
								) : (
									<>
										{/* Provider + model */}
										<div className="grid gap-4 md:grid-cols-2">
											<div className="space-y-2">
												<FieldLabel htmlFor="semantic-provider">Embedding provider</FieldLabel>
												<Controller
													control={control}
													name="semantic.provider"
													render={({ field }) => (
														<Select
															value={field.value || undefined}
															onValueChange={(value: ModelProviderName) => {
																if (value === field.value) return;
																field.onChange(value);
																// A model name, and the dimension measured from it, are
																// only meaningful for their own provider.
																setValue("semantic.embedding_model", "", { shouldDirty: true });
																setValue("semantic.dimension", 0, { shouldDirty: true });
																setDimensionProbeError(null);
															}}
															disabled={!canUpdate || noEmbeddingProviders}
														>
															<SelectTrigger
																className="w-full"
																id="semantic-provider"
																data-testid="complexity-router-semantic-provider-select"
															>
																<SelectValue placeholder="Select provider" />
															</SelectTrigger>
															<SelectContent>
																{embeddingProviders
																	.filter((provider) => provider.name)
																	.map((provider) => (
																		<SelectItem key={provider.name} value={provider.name}>
																			<div className="flex items-center gap-2">
																				<RenderProviderIcon provider={provider.name as ProviderIconType} size="sm" className="h-4 w-4" />
																				<span>{getProviderLabel(provider.name)}</span>
																			</div>
																		</SelectItem>
																	))}
															</SelectContent>
														</Select>
													)}
												/>
												{semanticErrors?.provider && <p className="text-destructive text-xs">{semanticErrors.provider.message}</p>}
											</div>

											<div className="space-y-2">
												<FieldLabel htmlFor="semantic-embedding-model">Embedding model</FieldLabel>
												<Controller
													control={control}
													name="semantic.embedding_model"
													render={({ field }) => (
														<ModelMultiselect
															inputId="semantic-embedding-model"
															data-testid="complexity-router-semantic-model-select"
															isSingleSelect
															provider={liveSemantic?.provider || undefined}
															value={field.value ?? ""}
															onChange={(model) => {
																field.onChange(model);
																void runDimensionProbe(liveSemantic?.provider ?? "", model);
															}}
															placeholder={liveSemantic?.provider ? "Search or type an embedding model…" : "Select a provider first"}
															disabled={!canUpdate || !liveSemantic?.provider}
														/>
													)}
												/>
												{/* The embedding dimension is measured from this model rather than
										    entered, so its probe reports here instead of in a field of its own. */}
												{isProbingDimension ? (
													<p className="text-muted-foreground flex items-center gap-1.5 text-xs">
														<LoaderCircle className="size-3 animate-spin" />
														Detecting embedding dimension…
													</p>
												) : dimensionProbeError ? (
													<p
														className="text-destructive flex flex-wrap items-center gap-1.5 text-xs"
														data-testid="complexity-router-semantic-dimension-error"
													>
														<span>{dimensionProbeError}</span>
														<Button
															data-testid="complexity-router-semantic-dimension-retry-button"
															type="button"
															variant="link"
															size="sm"
															className="h-auto p-0 text-xs"
															disabled={!canUpdate}
															onClick={() => runDimensionProbe(liveSemantic?.provider ?? "", liveSemantic?.embedding_model ?? "")}
														>
															Retry
														</Button>
													</p>
												) : semanticErrors?.embedding_model ? (
													<p className="text-destructive text-xs">{semanticErrors.embedding_model.message}</p>
												) : null}
											</div>
										</div>

										{/* Similarity floor + conversation window */}
										<div className="grid gap-4 md:grid-cols-2">
											<div className="space-y-2">
												<FieldLabel htmlFor="semantic-min-similarity">Minimum similarity threshold</FieldLabel>
												<Input
													id="semantic-min-similarity"
													data-testid="complexity-router-semantic-min-similarity-input"
													type="number"
													min={0}
													max={0.99}
													step={0.05}
													disabled={!canUpdate || !isClassifierConfigured}
													aria-invalid={semanticErrors?.min_similarity ? true : undefined}
													className={cn("font-mono", semanticErrors?.min_similarity && "border-destructive focus-visible:ring-destructive")}
													{...register("semantic.min_similarity", { valueAsNumber: true })}
												/>
												{semanticErrors?.min_similarity ? (
													<p className="text-destructive text-xs">{semanticErrors.min_similarity.message}</p>
												) : (
													<>
														<p className="text-muted-foreground text-xs leading-relaxed">
															Between 0 and 1. How close the nearest phrase must be before its tier is used.
														</p>
														<p className="flex items-start gap-1.5 text-xs leading-relaxed text-amber-700 dark:text-amber-400">
															<TriangleAlert className="mt-0.5 size-3 shrink-0" />
															Lower values raise the risk of false positives.
														</p>
													</>
												)}
											</div>

											<div className="space-y-2">
												<FieldLabel
													htmlFor="semantic-message-history"
													tooltip={
														<>
															The most recent user messages are joined oldest to newest and embedded as one text. Widening this lets a short
															follow-up like &ldquo;and make it faster&rdquo; inherit earlier intent, but dilutes the latest message and
															embeds more tokens. System prompts and assistant replies are never embedded.
														</>
													}
												>
													Max messages to embed
												</FieldLabel>
												<Input
													id="semantic-message-history"
													data-testid="complexity-router-semantic-message-history-input"
													type="number"
													min={MIN_SEMANTIC_MESSAGE_HISTORY}
													max={MAX_SEMANTIC_MESSAGE_HISTORY}
													step={1}
													disabled={!canUpdate || !isClassifierConfigured}
													aria-invalid={semanticErrors?.message_history_count ? true : undefined}
													className={cn(
														"font-mono",
														semanticErrors?.message_history_count && "border-destructive focus-visible:ring-destructive",
													)}
													{...register("semantic.message_history_count", { valueAsNumber: true })}
												/>
												{semanticErrors?.message_history_count && (
													<p className="text-destructive text-xs">{semanticErrors.message_history_count.message}</p>
												)}
											</div>
										</div>

										{/* Timeout + phrase storage */}
										<div className="grid gap-4 md:grid-cols-2">
											<div className="space-y-2">
												<FieldLabel
													htmlFor="semantic-timeout"
													tooltip="Ceiling on the embedding call, which runs inline on the request path. Exceeding it skips complexity tier based routing for that request."
												>
													Embedding timeout (ms)
												</FieldLabel>
												<Controller
													control={control}
													name="semantic.timeout"
													render={({ field }) => (
														<Input
															id="semantic-timeout"
															data-testid="complexity-router-semantic-timeout-input"
															type="number"
															min={1}
															step={10}
															disabled={!canUpdate || !isClassifierConfigured}
															value={semanticTimeoutFieldValue(field.value)}
															onChange={(event) => {
																const raw = event.target.value;
																field.onChange(raw === "" ? "" : `${raw}ms`);
															}}
															aria-invalid={semanticErrors?.timeout ? true : undefined}
															className={cn("font-mono", semanticErrors?.timeout && "border-destructive focus-visible:ring-destructive")}
														/>
													)}
												/>
												{semanticErrors?.timeout && <p className="text-destructive text-xs">{semanticErrors.timeout.message}</p>}
											</div>

											<div className="space-y-2">
												<FieldLabel
													htmlFor="semantic-vector-store"
													tooltip={
														<span className="space-y-1.5">
															{SEMANTIC_VECTOR_STORE_OPTIONS.map((option) => (
																<span key={option.value} className="block">
																	<b>{option.label}</b>: {option.tooltip}
																</span>
															))}
														</span>
													}
												>
													Reference phrase storage
												</FieldLabel>
												<Controller
													control={control}
													name="semantic.vector_store"
													render={({ field }) => (
														<Select
															value={field.value ?? "embedded"}
															onValueChange={field.onChange}
															disabled={!canUpdate || !isClassifierConfigured}
														>
															<SelectTrigger
																className="w-full"
																id="semantic-vector-store"
																data-testid="complexity-router-semantic-vector-store-select"
															>
																<SelectValue />
															</SelectTrigger>
															<SelectContent>
																{SEMANTIC_VECTOR_STORE_OPTIONS.map((option) => (
																	<SelectItem key={option.value} value={option.value}>
																		{option.label}
																	</SelectItem>
																))}
															</SelectContent>
														</Select>
													)}
												/>
												{liveSemantic?.vector_store === "auto" && !isVectorStoreConnected && (
													<p className="text-muted-foreground text-xs leading-relaxed">
														No vector store is connected, so phrases stay in the embedded store until one is configured.
													</p>
												)}
											</div>
										</div>

										{/* Budget attribution */}
										<div className="flex items-center justify-between gap-6 border-t pt-4">
											<FieldLabel
												htmlFor="semantic-count-toward-budgets"
												tooltip="Bills each classification embedding to the same budgets as the request that triggered it, and warmup embeddings to the provider and model budgets. Cost is always reported to telemetry either way."
											>
												Count embedding cost toward budgets
											</FieldLabel>
											<Controller
												control={control}
												name="semantic.count_toward_budgets"
												render={({ field }) => (
													<Switch
														id="semantic-count-toward-budgets"
														data-testid="complexity-router-semantic-budgets-switch"
														checked={field.value ?? false}
														onCheckedChange={field.onChange}
														disabled={!canUpdate || !isClassifierConfigured}
													/>
												)}
											/>
										</div>
									</>
								)}
							</div>
						</div>

						{/* ── Submit error ── */}
						{submitError && (
							<div
								role="alert"
								className="border-destructive/40 bg-destructive/10 text-destructive rounded-sm border px-3 py-2 font-mono text-sm"
							>
								{submitError}
							</div>
						)}
					</div>
				</ScrollArea>

				{/* ── Action footer ── */}
				<div className="bg-card border-t px-14 py-4">
					<div className="mx-auto flex w-full max-w-7xl flex-wrap items-center justify-end gap-2.5">
						<Button
							data-testid="complexity-router-restore-defaults-button"
							type="button"
							variant="ghost"
							size="sm"
							onClick={() => setRestoreDialogOpen(true)}
							disabled={!canUpdate || isSaving || isResetting}
						>
							{isResetting ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <RotateCcw className="h-3.5 w-3.5" />}
							Restore defaults
						</Button>
						<Button
							data-testid="complexity-router-discard-changes-button"
							type="button"
							variant="outline"
							size="sm"
							onClick={handleDiscard}
							disabled={!isDirty || isSaving || isResetting || isFetching}
						>
							Discard changes
						</Button>
						<Button
							data-testid="complexity-router-save-changes-button"
							type="submit"
							size="sm"
							disabled={!canUpdate || !isDirty || isSaving || isResetting || (isSubmitted && hasErrors)}
						>
							{isSaving ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
							{isSaving ? "Saving…" : "Save changes"}
						</Button>
					</div>
				</div>
			</form>

			<AlertDialog open={restoreDialogOpen} onOpenChange={setRestoreDialogOpen}>
				<AlertDialogContent>
					<AlertDialogHeader>
						<AlertDialogTitle>Restore defaults</AlertDialogTitle>
						<AlertDialogDescription>
							This will replace the phrase to tier mapping with the default reference phrases and clear the embedding configuration, turning
							classification off. Your current configuration will be lost. This action cannot be undone.
						</AlertDialogDescription>
					</AlertDialogHeader>
					<AlertDialogFooter>
						<AlertDialogCancel
							data-testid="complexity-router-restore-cancel-button"
							onClick={() => setRestoreDialogOpen(false)}
							disabled={isResetting}
						>
							Cancel
						</AlertDialogCancel>
						<AlertDialogAction
							data-testid="complexity-router-restore-confirm-button"
							onClick={() => {
								setRestoreDialogOpen(false);
								handleRestoreDefaults();
							}}
							disabled={!canUpdate || isResetting}
						>
							Restore defaults
						</AlertDialogAction>
					</AlertDialogFooter>
				</AlertDialogContent>
			</AlertDialog>
		</>
	);
}