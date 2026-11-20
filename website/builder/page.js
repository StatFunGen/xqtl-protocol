/* xQTL Analysis Workflow Builder.
 *
 * Source: website/builder/page.html (template), this file (logic) and
 * website/builder/page_content.json (method choices, module intros, example outputs).
 * website/builder_registry.py --write assembles them, with the commands, inputs and
 * outputs it reads from the module pages, into code/SoS/xqtl_protocol_workflow_builder.ipynb.
 * Edit these sources and re-run --write; never edit the notebook by hand.
 *
 * XQB_DATA (inserted by the registry):
 *   cmd          {"module|workflow": command}            from the module pages' builder cells
 *   io           {"module|workflow": {i: [...], o: [...]}} declared inputs and outputs
 *   untested     ["module|workflow", ...] not run on the example data, with untestedWhy reasons
 *   prev         {"tests/fixtures/...": {rows, cols}}     first rows of each example input file
 *   meth, intro, bench, outPrev                            from page_content.json
 */
(function () {
'use strict';
// R: where the builder's elements live. The standalone page uses the document; on the site's
// builder page the MyST widget renders it inside a shadow root, which sets XQB_ROOT first.
var R = (typeof XQB_ROOT !== 'undefined' && XQB_ROOT) || document;
var WIDGET = R !== document;
var XQB_DATA = JSON.parse(R.getElementById('xqb-data').textContent);
var REV = '__XQTL_PROTOCOL_REV__'.indexOf('__') === 0 ? 'unreleased' : '__XQTL_PROTOCOL_REV__';

/* ---- Rendering: a small renderer for the template's {{path}}, sc-for and sc-if ---- */
var EVENTS = { onclick: 'click', onchange: 'change', oninput: 'input' };
var PROPS = { checked: 1, disabled: 1, value: 1, selected: 1 };
function lookup(path, scope) {
  path = path.trim();
  if (path === 'true') return true;
  if (path === 'false') return false;
  var parts = path.split('.'), v;
  for (var i = scope.length - 1; i >= 0; i--) { if (parts[0] in scope[i]) { v = scope[i][parts[0]]; break; } }
  for (var j = 1; j < parts.length && v != null; j++) v = v[parts[j]];
  return v;
}
var ONE = /^\{\{([^}]+)\}\}$/, ANY = /\{\{([^}]+)\}\}/g;
function interp(text, scope) { return text.replace(ANY, function (_, p) { var v = lookup(p, scope); return v == null ? '' : String(v); }); }
var SVG = 'http://www.w3.org/2000/svg';
function build(node, scope, out, inSvg) {
  if (node.nodeType === 3) { out.appendChild(document.createTextNode(interp(node.nodeValue, scope))); return; }
  if (node.nodeType !== 1) return;
  var tag = node.tagName.toLowerCase();
  if (tag === 'sc-for') {
    var list = lookup(node.getAttribute('list').match(ONE)[1], scope) || [], as = node.getAttribute('as');
    list.forEach(function (item) { var s = {}; s[as] = item; kids(node, scope.concat([s]), out, inSvg); });
    return;
  }
  if (tag === 'sc-if') { if (lookup(node.getAttribute('value').match(ONE)[1], scope)) kids(node, scope, out, inSvg); return; }
  inSvg = inSvg || tag === 'svg';
  var el = inSvg ? document.createElementNS(SVG, node.tagName) : document.createElement(tag);
  var late = [];
  Array.prototype.forEach.call(node.attributes, function (a) {
    var name = a.name, m = a.value.match(ONE), lname = name.toLowerCase();
    if (EVENTS[lname]) { var fn = m && lookup(m[1], scope); if (typeof fn === 'function') el.addEventListener(EVENTS[lname], fn); return; }
    if (PROPS[lname] && m) { late.push([lname, lookup(m[1], scope)]); return; }
    el.setAttribute(name, interp(a.value, scope));
  });
  kids(node, scope, el, inSvg);
  late.forEach(function (p) { if (p[0] === 'value') { el.value = p[1] == null ? '' : p[1]; el.setAttribute('value', el.value); } else { el[p[0]] = !!p[1]; if (p[1]) el.setAttribute(p[0], ''); } });
  out.appendChild(el);
}
function kids(node, scope, out, inSvg) { var c = node.childNodes; for (var i = 0; i < c.length; i++) build(c[i], scope, out, inSvg); }

function copyToClipboard(text, done) {
  function fallback() {
    var t = document.createElement('textarea'); t.value = text; t.setAttribute('readonly', ''); t.style.position = 'fixed'; t.style.opacity = '0';
    document.body.appendChild(t); t.select(); var ok = false; try { ok = document.execCommand('copy'); } catch (e) {} document.body.removeChild(t); done(ok);
  }
  try { navigator.clipboard.writeText(text).then(function () { done(true); }, fallback); } catch (e) { fallback(); }
}

/* ---- Content and route logic (from the approved design) ---- */
var BASE = 'https://statfungen.github.io/xqtl-protocol/';
var DOC = {"reference_data_preparation":"reference-data-preparation","generalized_TADB":"generalized-tadb","ld_prune_reference":"ld-prune-reference","rss_ld_sketch":"rss-ld-sketch","RNA_calling":"rna-calling","bulk_expression_QC":"bulk-expression-qc","bulk_expression_normalization":"bulk-expression-normalization","snRNAseq_preprocessing":"snrnaseq-preprocessing","pseudobulk_preprocessing":"pseudobulk-preprocessing","splicing_calling":"splicing-calling","splicing_calling_leafcutter2":"splicing-calling-leafcutter2","isoform_calling_torino":"isoform-calling-torino","splicing_normalization":"splicing-normalization","methylation_calling":"methylation-calling","apa_calling":"apa-calling","apa_impute":"apa-impute","VCF_QC":"vcf-qc","genotype_formatting":"genotype-formatting","GWAS_QC":"gwas-qc","PCA":"pca","gene_annotation":"gene-annotation","phenotype_imputation":"phenotype-imputation","phenotype_formatting":"phenotype-formatting","covariate_formatting":"covariate-formatting","covariate_hidden_factor":"covariate-hidden-factor","TensorQTL":"tensorqtl","qr_and_twas":"qr-and-twas","qtl_association_postprocessing":"qtl-association-postprocessing","mash_preprocessing":"mash-preprocessing","mixture_prior":"mixture-prior","mash_fit":"mash-fit","mash_posterior":"mash-posterior","mnm_regression":"mnm-regression","rss_analysis":"rss-analysis","qtl_rss_analysis":"qtl-rss-analysis","SuSiE_enloc":"susie-enloc","twas_ctwas":"twas-ctwas","colocboost":"colocboost","eoo_enrichment":"eoo-enrichment","gsea":"gsea","gregor":"gregor","sldsc_enrichment":"sldsc-enrichment","ems_training":"ems-training","ems_prediction":"ems-prediction","METAL":"metal","watershed":"watershed","mnm_postprocessing":"mnm-postprocessing","intact":"intact"};
var CATALOG = [
  ['Reference data', 'Genome, annotation and LD panels every later stage uses', [['reference_data_preparation','Reference Preparation','Download and standardize references'],['generalized_TADB','Generalized TADB','TAD boundary annotations'],['ld_prune_reference','LD Reference Pruning','Pruned LD reference panels'],['rss_ld_sketch','RSS LD Sketching','LD sketches for summary-statistic methods']]],
  ['Molecular phenotypes', 'Turn raw assays into measured features', [['RNA_calling','Quantifying Expression','STAR + RNA-SeQC / RSEM'],['bulk_expression_QC','Sample-level QC','Outlier removal, QC metrics'],['bulk_expression_normalization','Counts Normalization','TMM / TPM normalization'],['snRNAseq_preprocessing','snRNA-seq Preprocessing','Cell QC, clustering, annotation'],['pseudobulk_preprocessing','Pseudobulk Preprocessing','Aggregation for RNA and ATAC'],['splicing_calling','Splicing Calling','Intron cluster identification'],['splicing_calling_leafcutter2','Splicing Calling (LeafCutter2)','Intron clusters with productive and unproductive labels'],['isoform_calling_torino','Isoform Usage (Torino)','Isoform usage from read coverage'],['splicing_normalization','Splicing Normalization','Normalization and filtering'],['methylation_calling','Methylation Calling','Array-based pipeline'],['apa_calling','APA Calling','Polyadenylation site calling'],['apa_impute','APA Imputation & QC','Missing value imputation']]],
  ['Genotype and phenotype preparation', 'QC, formatting and covariates', [['VCF_QC','VCF QC','Variant filters on raw VCF'],['genotype_formatting','Genotype Formatting','VCF to PLINK, merge, split by chromosome'],['GWAS_QC','Genotype QC','PLINK QC + KING kinship'],['PCA','PCA','Genetic PCs on unrelated samples'],['gene_annotation','Gene Annotation','Attach genomic coordinates'],['phenotype_imputation','Phenotype Imputation','Impute missing values'],['phenotype_formatting','Phenotype Formatting','Per-chromosome BED for TensorQTL'],['covariate_formatting','Covariate Formatting','Merge PCs with covariates'],['covariate_hidden_factor','Hidden Factors','Marchenko PCA / PEER']]],
  ['QTL association testing', 'Find where genotype tracks the phenotype', [['TensorQTL','TensorQTL','Cis, interaction and trans scans'],['qtl_association_postprocessing','Post-processing','Hierarchical multiple testing'],['qr_and_twas','Quantile Regression QTL','Quantile QTL + quantile TWAS weights'],['METAL','METAL','Meta-analysis across cohorts']]],
  ['Multi-context sharing and fine-mapping', 'Shared effects and credible sets', [['mash_preprocessing','MASH Preprocessing','Prepare effect sizes and SEs'],['mixture_prior','Mixture Prior','Data-driven covariances'],['mash_fit','MASH Fitting','Fit model and posteriors'],['mash_posterior','MASH Posterior','Posterior effect summaries'],['mnm_regression','Fine-mapping and TWAS weights','SuSiE, mvSuSiE, fSuSiE, multi-gene'],['mnm_postprocessing','Fine-mapping post-processing','Export results and top loci'],['qtl_rss_analysis','QTL summary-statistic fine-mapping','SuSiE-RSS on xQTL summary statistics'],['rss_analysis','GWAS summary-statistic fine-mapping','SuSiE-RSS on GWAS summary statistics']]],
  ['GWAS integration', 'Relate regulation to complex traits', [['SuSiE_enloc','SuSiE-enloc','Pairwise xQTL-GWAS colocalization'],['twas_ctwas','TWAS & cTWAS','Gene-level trait association'],['colocboost','ColocBoost','Shared variants across many traits'],['intact','INTACT','Combine TWAS and colocalization evidence'],['watershed','Watershed','Rare-variant functional prediction']]],
  ['Enrichment and scoring', 'Which annotations and pathways are involved', [['eoo_enrichment','Excess of Overlap (EOO)','Block-jackknife enrichment'],['gsea','GSEA','Pathway overrepresentation'],['gregor','GREGOR','SNP set vs matched null'],['sldsc_enrichment','S-LDSC','Heritability partitioning'],['ems_training','scEEMS Training','Fit per-variant CatBoost model'],['ems_prediction','scEEMS Prediction','Score variants with a trained model']]]
];
// The overview flowchart: phases (top to bottom) > boxes > groups > modules, as on the original builder.
// The eight phases of the original builder (FLOW below holds their module groups).
// [purpose, when to use, build preset or null]; "when" is phase- or stream-level text, since the
// module pages carry no "when to use" field yet.
var PHASES = [
  ['Genome, annotations and LD panels, built once and read where later stages need them', 'When a later step needs a reference genome, gene annotation or LD panel you do not already have.', null],
  ['Turn raw assays into measured traits; pick the branches your study has', 'When you start from raw sequencing or array data rather than a phenotype matrix.', { label: 'Build a route that starts from raw assay data', study: { start: 'raw', goals: { discover: true } } }],
  ['Genotype, phenotype and covariate preparation, side by side', '', { label: 'Build a route that starts from raw genotypes (VCF)', study: { geno: 'vcf', goals: { discover: true } } }],
  ['Association scan (cis, interaction or trans), then multiple-testing correction', 'In every QTL study: this is where variants are tested against each molecular trait.', { label: 'Build a QTL discovery route', example: 'eqtl', study: { goals: { discover: true } } }],
  ['Combine evidence across cohorts (METAL) or across tissues and cell types (MASH)', 'When you have several cohorts to combine, or several contexts whose effects you want to compare.', { label: 'Build a route that compares contexts', study: { contexts: 'multiple', goals: { discover: true, mash: true } } }],
  ['Prioritize candidate causal variants within a signal; a credible set is not proof of causality', 'When you want to narrow an association to the variants most likely to drive it, from individual-level data or from summary statistics with an LD reference.', { label: 'Build a fine-mapping route', example: 'finemap', study: { goals: { discover: true, finemap: true } } }],
  ['Colocalization (a shared signal, not proof of mediation) or TWAS (predicted gene level vs the trait)', 'When you want to connect molecular QTLs to a disease or complex-trait GWAS.', { label: 'Build a GWAS integration route', example: 'gwas', study: { goals: { discover: true, finemap: true, gwas: true } } }],
  ['Annotation, pathway and heritability enrichment, and variant scoring', 'When you want to interpret QTLs or credible sets with functional annotations; methods answer different questions and none is required.', { label: 'Build a route that tests enrichment', study: { goals: { discover: true, finemap: true, enrich: true } } }]
];
var STREAMS = [
  ['Genotype', 'QC, formatting and genetic PCs', 'Before association testing, to quality-control genotypes and compute genetic PCs for population structure.'],
  ['Phenotype', 'Annotation, imputation and per-chromosome files', 'Before association testing, to place each trait on the genome and fill or format the phenotype matrix.'],
  ['Covariate', 'Known covariates and hidden factors', 'When unmeasured technical or biological variation may confound association testing; estimate factors before the scan and evaluate their effects.']
];
var OPTIONAL = { 4: 1, 5: 1, 6: 1, 7: 1 };
var FLOW = [
  ['Inputs', 1, [['Reference Data', [[null, ['reference_data_preparation', 'generalized_TADB', 'ld_prune_reference', 'rss_ld_sketch']]]]]],
  ['Quantification', 2, [['Molecular Phenotype Quantification', [['Bulk RNA-seq', ['RNA_calling', 'bulk_expression_QC', 'bulk_expression_normalization']], ['Single-nuclei / pseudobulk', ['snRNAseq_preprocessing', 'pseudobulk_preprocessing']], ['Alternative splicing', ['splicing_calling', 'splicing_calling_leafcutter2', 'isoform_calling_torino', 'splicing_normalization']], ['DNA methylation', ['methylation_calling']], ['Polyadenylation', ['apa_calling', 'apa_impute']]]]]],
  ['Pre-processing', 3, [['Data Pre-processing', [['Genotype', ['VCF_QC', 'genotype_formatting', 'GWAS_QC', 'PCA']], ['Phenotype', ['gene_annotation', 'phenotype_imputation', 'phenotype_formatting']], ['Covariate', ['covariate_formatting', 'covariate_hidden_factor']]]]]],
  ['Discovery', 4, [['QTL Association Testing', [[null, ['TensorQTL', 'qtl_association_postprocessing', 'qr_and_twas']]]]]],
  ['Multivariate modeling', 5, [['Cross-cohort Meta-analysis', [[null, ['METAL']]]], ['Multivariate Mixture (MASH)', [[null, ['mash_preprocessing', 'mixture_prior', 'mash_fit', 'mash_posterior']]]]]],
  ['Regression', 6, [['High-dimensional Regression', [['Individual level', ['mnm_regression', 'mnm_postprocessing']], ['Summary statistics level', ['qtl_rss_analysis', 'rss_analysis']]]]]],
  ['Integration', 7, [['GWAS Integration', [[null, ['SuSiE_enloc', 'twas_ctwas', 'colocboost', 'intact']]]], ['Rare-variant xQTL', [[null, ['watershed']]]]]],
  ['Interpretation', 8, [['Enrichment & Validation', [[null, ['eoo_enrichment', 'gsea', 'gregor', 'sldsc_enrichment']]]], ['xQTL Modifier Score', [[null, ['ems_training', 'ems_prediction']]]]]]
];
// The stage map: parallel inputs meet at QTL discovery, then branch by question.
var SOON = {METAL: 1, watershed: 1, ems_prediction: 1, qr_and_twas: 1};
var LABEL = {}, STAGE_OF = {}, DESC = {};
CATALOG.forEach(function (g, gi) { g[2].forEach(function (m) { LABEL[m[0]] = m[1]; STAGE_OF[m[0]] = gi; DESC[m[0]] = m[2]; }); });
var FIXED_CHOICES = {
  TensorQTL: [['cis', 'Cis scan', true, 'Tests variants near each gene. The protocol’s main analysis.'], ['cis:interaction', 'Interaction scan', false, 'Cis scan with a genotype × covariate term, for example sex.'], ['trans', 'Trans scan', false, 'Tests variants far from each gene, on other chromosomes too.']],
  RNA_calling: [['rnaseqc_call', 'RNA-SeQC gene counts', true, 'Gene-level read counts and QC metrics.'], ['rsem_call', 'RSEM', false, 'Transcript and gene expression estimates.']],
  methylation_calling: [['sesame', 'SeSAMe', true, 'Beta values with SeSAMe normalization.'], ['minfi', 'minfi', false, 'Beta values with quantile normalization; needs several samples.']]
};
var STUDY0 = { ph: 'bulk', start: 'matrix', goals: { discover: true, finemap: false, gwas: false, mash: false, enrich: false }, geno: 'plink_qc', cohorts: 'single', contexts: 'single', quantile: 'no', fmfrom: 'indiv', link: 'colocboost', enrichq: 'eoo' };
var MAIN = {
  TensorQTL: ['Variant–phenotype association results', 'One row per variant and molecular trait tested near each gene: effect size, standard error and p-value. These are the raw cis-QTL associations.', 'Read the post-processed table first; use these when you need every tested pair.'],
  qtl_association_postprocessing: ['Significant QTLs after multiple-testing correction', 'Genes with a QTL after correcting across variants and genes, and the variants that pass. This is the list most studies report.', 'Fine-map the significant genes, or compare them across contexts.'],
  mnm_regression: ['Fine-mapping credible sets and TWAS weights', 'Credible sets: small groups of variants that together contain a causal variant with high probability, plus per-variant inclusion probabilities. The same run stores the weights TWAS uses.', 'Connect the credible sets to a GWAS trait by colocalization or TWAS.'],
  qtl_rss_analysis: ['Fine-mapping from summary statistics', 'Credible sets estimated from summary statistics and an LD reference.', 'Connect them to a GWAS trait.'],
  colocboost: ['Colocalization across traits', 'Variants shared between the molecular trait and the GWAS trait, with the probability that they share one causal signal.', 'Check the shared variants against annotation, or test enrichment.'],
  SuSiE_enloc: ['Pairwise colocalization', 'For each gene and GWAS region, the probability that the QTL and the trait share a causal variant.', 'Prioritize genes with a high shared-signal probability.'],
  twas_ctwas: ['Gene-level trait association (TWAS)', 'For each gene, whether its genetically predicted level is associated with the trait.', 'Follow significant genes with colocalization to check they share a signal.'],
  mash_posterior: ['Shared and context-specific effects', 'Effects re-estimated across contexts, showing which QTLs are shared and which are specific.', 'Fine-map context-specific QTLs.'],
  eoo_enrichment: ['Annotation enrichment', 'Whether your variants fall in an annotation more often than expected.', 'Report the enriched annotations.'],
  gsea: ['Pathway enrichment', 'Pathways over-represented among your genes.', 'Report the enriched pathways.'],
  gregor: ['SNP-set enrichment', 'Whether your SNP set overlaps an annotation more than matched SNPs.', 'Report the enriched annotations.'],
  sldsc_enrichment: ['Heritability enrichment', 'How much trait heritability an annotation explains relative to its size.', 'Report the enriched annotations.']
};
var REF_RX = /(^reference_data\/|ld_reference|\/reference\/|\.gtf|\.fa(sta)?(\.gz)?$|ld_meta|LD_blocks|_Index)/i;
// replace one declared input path, as a whole word, in a command
function swapPath(c, p, to) { var esc = p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); return c.replace(new RegExp('(^|[\\s=\'"`])' + esc + '(?=$|[\\s\'"`;|)])', 'g'), function (_, pre) { return pre + to; }); }
var EXAMPLES = [
  { id: 'eqtl', stage: 3, label: 'eQTL discovery, starting from prepared data', study: { goals: { discover: true } }, title: 'Example: discover eQTLs', learn: 'How genotypes and gene expression are combined to find variants associated with each gene’s expression.', given: 'Genotypes on chromosomes 21 and 22, plus an expression matrix and covariates for the same small set of samples. The expression data cover chromosome 22, so association is tested there.', result: 'Variant–expression association results and the list of genes with a significant eQTL.', after: 'Open the results table, then use the same workflow with your own data.' },
  { id: 'finemap', stage: 4, label: 'Fine-mapping, from the same prepared data', study: { goals: { discover: true, finemap: true } }, title: 'Example: fine-map eQTL signals', learn: 'How fine-mapping narrows a QTL signal to credible sets of likely causal variants.', given: 'The same example genotypes, expression and covariates.', result: 'Association results, then credible sets and TWAS weights for an example gene.', after: 'Look at the credible sets, then connect them to a GWAS trait.' },
  { id: 'gwas', stage: 5, label: 'GWAS integration, colocalization with an example trait', study: { goals: { discover: true, finemap: true, gwas: true }, link: 'colocboost' }, title: 'Example: connect eQTLs to a GWAS trait', learn: 'How colocalization asks whether a QTL and a trait association share a causal variant.', given: 'The example QTL data plus example GWAS summary statistics and an LD reference.', result: 'Colocalized variants shared by the gene and the trait.', after: 'Prioritize the shared genes, or test enrichment.' }
];
function clone(o) { return JSON.parse(JSON.stringify(o)); }
function studyFor(ex) { var s = clone(STUDY0); var e = EXAMPLES.filter(function (x) { return x.id === ex; })[0] || EXAMPLES[0]; Object.keys(e.study).forEach(function (k) { s[k] = k === 'goals' ? Object.assign(clone(STUDY0.goals), e.study.goals) : e.study[k]; }); return s; }

function buildRoute(s, methods, D) {
  var R = [], meth = (D && D.meth) || {};
  if (!s || !s.ph && s.start !== 'sumstats') return R;
  function choicesFor(m) { return meth[m] || FIXED_CHOICES[m] || null; }
  function add(id, m, wfs, title, opt) { R.push(Object.assign({ id: id, m: m, wfs: wfs, title: title }, opt || {})); }
  function addChoice(id, m, title, def) { var ch = choicesFor(m); var d = def || (ch ? (ch.filter(function (c) { return c[2]; })[0] || ch[0])[0] : null); var w = (methods && methods[id]) || d; add(id, m, [w], title, { choices: ch, chosen: w, deflt: d }); }
  var sum = s.start === 'sumstats', g = s.goals;
  if (s.start === 'raw') {
    if (s.ph === 'bulk') { addChoice('quant', 'RNA_calling', 'Quantify expression'); add('bqc', 'bulk_expression_QC', ['qc'], 'Remove outlier samples'); }
    if (s.ph === 'splice') { add('spl', 'splicing_calling_leafcutter2', ['leafcutter2'], 'Call splicing events'); add('spn', 'splicing_normalization', ['leafcutter_norm'], 'Normalize splicing ratios'); }
    if (s.ph === 'meth') addChoice('mc', 'methylation_calling', 'Call methylation');
  }
  if (!sum && s.ph) {
    if (s.geno === 'vcf') add('vqc', 'VCF_QC', ['qc'], 'Variant QC on the raw VCF');
    if (s.geno !== 'plink_qc') { add('gf', 'genotype_formatting', ['merge_plink'], 'Merge genotype files'); add('gqc', 'GWAS_QC', ['qc_no_prune', 'genotype_phenotype_sample_overlap', 'king', 'qc'], 'Genotype QC and relatedness'); add('pca', 'PCA', ['flashpca'], 'Genotype principal components'); }
    add('gbc', 'genotype_formatting', ['genotype_by_chrom'], 'Split genotypes by chromosome');
    add('pf', 'phenotype_formatting', ['phenotype_by_chrom'], 'Phenotype files per chromosome');
    add('cf', 'covariate_formatting', ['merge_genotype_pc'], 'Merge covariates with genotype PCs');
    addChoice('hf', 'covariate_hidden_factor', 'Estimate hidden factors');
    if (g.discover || g.mash || s.quantile === 'interaction') { addChoice('tq', 'TensorQTL', 'Association scan', s.quantile === 'interaction' ? 'cis:interaction' : null); add('pp', 'qtl_association_postprocessing', ['default'], 'Multiple-testing correction'); }
    if (s.cohorts === 'multiple') add('metal', 'METAL', ['METAL'], 'Meta-analysis across cohorts', { soon: true });
    if (g.mash) { add('mp', 'mash_preprocessing', ['random_null_tensorqtl'], 'Prepare effects for MASH'); add('mx', 'mixture_prior', ['ed_bovy'], 'Learn sharing patterns'); add('mf', 'mash_fit', ['mash'], 'Fit MASH'); add('mpo', 'mash_posterior', ['posterior'], 'Shared and specific effects'); }
  }
  if (g.finemap) { if (sum || s.fmfrom === 'sumstats') add('fm', 'qtl_rss_analysis', ['qtl_rss'], 'Fine-map from summary statistics'); else addChoice('fm', 'mnm_regression', 'Fine-map and build TWAS weights', s.contexts === 'multiple' ? 'mnm' : null); }
  if (g.gwas) { var L = { colocboost: ['colocboost', ['colocboost'], 'Colocalize with GWAS'], enloc: ['SuSiE_enloc', ['susie_coloc'], 'Colocalize with GWAS'], twas: ['twas_ctwas', ['twas'], 'Gene-level association (TWAS)'] }[s.link] || ['colocboost', ['colocboost'], 'Colocalize with GWAS']; add('gw', L[0], L[1], L[2]); }
  if (g.enrich) { var E = { eoo: ['eoo_enrichment', ['enrichment']], gsea: ['gsea', ['pathway_analysis']], gregor: ['gregor', ['gregor']], sldsc: ['sldsc_enrichment', ['get_heritability']] }[s.enrichq]; add('en', E[0], E[1], 'Test enrichment'); }
  return closeRoute(R, D);
}
// Add, before each step, the builder steps that make the files it reads (and their own
// inputs), using the declared outputs: the same links the example-data tests follow. So a
// route always runs from start to finish, and a downloaded example script runs as is.
function closeRoute(R, D) {
  if (!D || !D.io) return R;
  var outs = [];
  Object.keys(D.io).forEach(function (k) { (D.io[k].o || []).forEach(function (o) { outs.push([k, o]); }); });
  var producer = function (p) { var best = null, len = -1; outs.forEach(function (ko) { var o = ko[1]; if ((p === o || p.indexOf(o + '/') === 0) && o.length > len) { best = ko[0]; len = o.length; } }); return best; };
  var placed = {};
  R.forEach(function (r) { r.wfs.forEach(function (w) { placed[r.m + '|' + w] = 1; }); });
  var res = [];
  function needs(k) {
    var io = D.io[k] || {}, ks = [];
    (io.i || []).forEach(function (p) { if (p.indexOf('output/') !== 0) return; var pk = ((D.src || {})[k] || {})[p] || producer(p); if (pk && pk !== k) ks.push(pk); });
    return ks.concat((D.after || {})[k] || []);
  }
  function place(k, forTitle) {
    if (placed[k]) return; placed[k] = 1;
    needs(k).forEach(function (pk) { place(pk, forTitle); });
    var mw = k.split('|');
    res.push({ id: 'dep-' + k.replace(/[^A-Za-z0-9]+/g, '-'), m: mw[0], wfs: [mw[1]], title: 'Prepare an input: ' + mw[1], auto: true });
  }
  R.forEach(function (r) { r.wfs.forEach(function (w) { needs(r.m + '|' + w).forEach(function (pk) { place(pk); }); }); res.push(r); });
  return res;
}
// "genoFile" -> "Genotypes (chr21.bed)": what the file is for, then its name
var OPT_NAMES = { genofile: 'Genotypes', phenofile: 'Phenotype data', covfile: 'Covariates', covariate_file: 'Covariates', pcafile: 'Genetic PCs', phenotype_file: 'Phenotype data', genotype_file: 'Genotypes', region_list: 'Regions', customized_association_windows: 'Association windows', gwas_meta_data: 'GWAS summary statistics list', ld_meta_data: 'LD reference list', annotation_file: 'Annotation' };
function optLabel(opt, p) {
  var base = p.replace(/^<|>$/g, '').split('/').pop();
  if (!opt) return base;
  var o = opt.replace(/^-+/, '').toLowerCase().replace(/-/g, '_');
  var name = OPT_NAMES[o] || opt.replace(/^-+/, '').replace(/[_-]+/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase().replace(/^./, function (c) { return c.toUpperCase(); });
  return name + ' (' + base + ')';
}
function sig(r) { return r.m + ':' + r.wfs.join(','); }
function diff(a, b) { var A = {}, ch = {}, n = 0; a.forEach(function (r) { A[r.id] = sig(r); }); b.forEach(function (r) { if (A[r.id] !== sig(r)) { ch[r.id] = 1; n++; } }); var B = {}; b.forEach(function (r) { B[r.id] = 1; }); a.forEach(function (r) { if (!B[r.id]) n++; }); return { changed: ch, text: n ? 'Route updated: ' + n + ' step' + (n > 1 ? 's' : '') + ' changed.' : '' }; }
// one option per line for `sos run` lines; setup lines (mkdir, cp, head) stay as written
function fmt(c, extra) {
  return c.split('\n').map(function (l) {
    if (!/^sos run /.test(l)) return l;
    var out = '', q = '';                      // break before options, never inside quotes or `...`
    for (var i = 0; i < l.length; i++) {
      var ch = l[i];
      if (q) { if (ch === q) q = ''; } else if (ch === '"' || ch === "'" || ch === '`') q = ch;
      else if (ch === ' ' && l[i + 1] === '-' && /[A-Za-z-]/.test(l[i + 2] || '')) { out += ' \\\n   '; }
      out += ch;
    }
    return out + (extra || '');
  }).join('\n');
}
function nice(p) { var b = p.split('/').pop(); return b.replace(/^protocol_example\./, '').replace(/\.(gz|bed|tsv|txt|csv|bgz)$/g, '').replace(/[._]+/g, ' '); }


// A shareable link: mode, example, study settings, methods and where it runs, in the URL hash.
// File paths are never put in the link.
function shareState(s) { return { m: s.mode, e: s.example, s: s.study, x: s.methods, r: s.runOn }; }
function shareUrl(s) { var base = 'https://statfungen.github.io/xqtl-protocol/xqtl_protocol_workflow_builder.html'; try { if (window.self === window.top && location.protocol.indexOf('http') === 0) base = location.href.split('#')[0]; } catch (e) {} return base + '#b=' + encodeURIComponent(JSON.stringify(shareState(s))); }
function readShared() {
  try { var h = (location.hash || '').match(/[#&]b=([^&]+)/); if (!h) return null; var o = JSON.parse(decodeURIComponent(h[1]));
    if (!o || typeof o !== 'object') return null;
    var st = clone(STUDY0); Object.keys(st).forEach(function (k) { if (o.s && o.s[k] != null) st[k] = k === 'goals' ? Object.assign(clone(STUDY0.goals), o.s.goals) : o.s[k]; });
    return { mode: o.m === 'own' ? 'own' : 'example', src: o.m === 'own' ? 'own' : 'example', example: EXAMPLES.some(function (e) { return e.id === o.e; }) ? o.e : 'eqtl', study: st, methods: o.x && typeof o.x === 'object' ? o.x : {}, runOn: ['local', 'slurm', 'sge'].indexOf(o.r) >= 0 ? o.r : 'local' };
  } catch (e) { return null; }
}
class Component {
  constructor() {
    this.state = { data: XQB_DATA, mode: 'example', example: 'eqtl', study: studyFor('eqtl'), methods: {}, src: 'example', runOn: 'local', paths: {}, mapView: 'all', selMod: '', open: {}, pv: {}, expStage: {}, openPh: {}, altPh: {}, phase: null, stream: null, catOpen: false, catOnly: null, catOnlyLabel: '', changed: {}, msg: '', adv: false, q: '', filter: 'all', copied: '' };
  }
  setRoute(patch, label) {
    var s = this.state, D = s.data;
    var nextStudy = patch.study || s.study, nextMethods = patch.methods || s.methods;
    var d = diff(buildRoute(s.study, s.methods, D), buildRoute(nextStudy, nextMethods, D));
    this.setState(Object.assign({}, patch, { changed: d.changed, msg: [label, d.text].filter(Boolean).join(' ') }));
  }
  renderVals() {
    var s = this.state, self = this, D = s.data || { cmd: {}, io: {}, untested: [], bench: {}, meth: {}, intro: {}, terms: {}, untestedWhy: {} };
    var untested = {}; (D.untested || []).forEach(function (k) { untested[k] = 1; });
    var testedKey = function (k) { return !!D.cmd[k] && !untested[k]; };
    // status as coloured text with a small dot, not a filled box (bg is kept for the call sites)
    var chip = function (bg, fg, extra) { return 'font-size: 13.5px; font-weight: 600; padding-left: 13px; color: ' + fg + '; background: radial-gradient(circle, ' + fg + ' 3.5px, transparent 4px) 0 50% / 8px 8px no-repeat;' + (extra || ''); };
    var tableOf = function (pv) { return { head: pv ? pv.rows[0].map(function (v) { return { v: v }; }) : [], rows: pv ? pv.rows.slice(1).map(function (row) { return { cells: row.map(function (v) { return { v: v }; }) }; }) : [] }; };
    var fileItem = function (key, p, extra) {
      var pv = (D.prev || {})[p], on = !!s.pv[key] && !!pv, t = tableOf(pv);
      return Object.assign({ label: nice(p).replace(/^./, function (c) { return c.toUpperCase(); }), path: p, hasPreview: !!pv, open: on, expanded: String(on), btn: on ? 'Hide example' : 'Show example',
        note: pv ? 'First ' + (pv.rows.length - 1) + ' rows' + (pv.cols > 6 ? ', 6 of ' + pv.cols + ' columns' : '') + ' of the example file.' : '', head: t.head, rows: t.rows,
        toggle: function () { var q = Object.assign({}, s.pv); q[key] = !q[key]; self.setState({ pv: q }); } }, extra || {});
    };
    var outOf = function (m) { var op = (D.outPrev || {})[m], t = tableOf(op); return { hasOut: !!op, outNote: op ? 'Example output, ' + op.file + ': first ' + (op.rows.length - 1) + ' rows' + (op.cols > 6 ? ', 6 of ' + op.cols + ' columns' : '') + '.' : '', outHead: t.head, outRows: t.rows }; };
    // outputs one per line; the example output's preview sits on the output it comes from
    var outList = function (m, outs) {
      var op = (D.outPrev || {})[m], ot = outOf(m), key = 'mo:' + m, on = !!s.pv[key];
      var at = op ? Math.max(0, outs.findIndex(function (o) { return op.file.indexOf(o.replace(/\/$/, '')) >= 0 || o.indexOf(op.file) >= 0; })) : -1;
      return outs.map(function (o, i) { var has = i === at;
        return { v: o, hasPreview: has, open: has && on, expanded: String(has && on), btn: has && on ? 'Hide example' : 'Show example', note: ot.outNote, head: ot.outHead, rows: ot.outRows,
          toggle: function () { var q = Object.assign({}, s.pv); q[key] = !q[key]; self.setState({ pv: q }); } }; });
    };
    var segStyle = function (on) { return 'font-size: 15px; min-height: 36px; padding: 6px 14px; border: 0; cursor: pointer; background: ' + (on ? 'var(--navy)' : 'var(--bg)') + '; color: ' + (on ? 'var(--bg)' : 'var(--ink)') + '; font-weight: ' + (on ? '600' : '400'); };
    var pillStyle = function (on) { return 'display: inline-flex; align-items: center; gap: 8px; font-size: 15px; min-height: 38px; padding: 7px 13px; border-radius: 6px; cursor: pointer; box-sizing: border-box; border: 1px solid ' + (on ? 'var(--navy)' : 'var(--ln2)') + '; background: ' + (on ? 'var(--acs)' : 'var(--bg)') + '; color: var(--ink); font-weight: ' + (on ? '600' : '400'); };

    var route = buildRoute(s.study, s.methods, s.data);
    var onRoute = {}, stepOf = {}; route.forEach(function (r) { onRoute[r.m] = 1; if (!stepOf[r.m]) stepOf[r.m] = r.id; });
    var routeOk = function (rt) { return rt.length > 0 && rt.every(function (r) { return !r.soon && r.wfs.every(function (w) { return testedKey(r.m + '|' + w); }); }); };

    // 1. Map
    var mapRoute = s.mapView === 'route' && route.length > 0;
    var map = CATALOG.map(function (g, gi) {
      var nOn = g[2].filter(function (m) { return onRoute[m[0]]; }).length;
      var active = !mapRoute || nOn > 0;
      var mods = g[2].filter(function (m) { return !mapRoute || onRoute[m[0]]; }).map(function (m) {
        var sel = s.selMod === m[0], on = !!onRoute[m[0]];
        return { label: m[1], pressed: String(sel), pick: function () { var e = Object.assign({}, s.expStage); e[gi] = true; self.setState({ selMod: sel ? '' : m[0], expStage: e }); },
          style: 'font-size: 14px; min-height: 34px; padding: 5px 11px; border-radius: 6px; cursor: pointer; border: 1px solid ' + (sel ? 'var(--navy)' : (on ? 'var(--teal)' : 'var(--ln)')) + '; background: ' + (sel ? 'var(--navy)' : (on ? 'var(--acs)' : 'var(--bg)')) + '; color: ' + (sel ? 'var(--bg)' : 'var(--ink)') + '; font-weight: ' + (on || sel ? '600' : '400') };
      });
      var exp = mapRoute || !!s.expStage[gi];
      return { n: gi + 1, label: g[0], sub: mapRoute ? nOn + ' module' + (nOn === 1 ? '' : 's') + ' on your route' : g[1], mods: mods,
        showMods: exp, expanded: String(exp), btn: mapRoute ? '' : (exp ? 'Hide modules' : 'Show ' + g[2].length + ' modules'),
        toggle: function () { var e = Object.assign({}, s.expStage); e[gi] = !e[gi]; self.setState({ expStage: e }); },
        titleStyle: 'font-size: 16px; font-weight: 600; color: ' + (active ? 'var(--navy)' : 'var(--ink2)'),
        badge: 'width: 28px; height: 28px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 700; ' + (active ? 'background: var(--navy); color: var(--on)' : 'background: var(--sk); color: var(--ink2)'),
        rail: gi < CATALOG.length - 1 ? 'flex: 1 1 auto; width: 2px; background: var(--ln); margin: 4px 0' : 'flex: 1 1 auto; width: 2px; background: transparent' };
    });
    var mapViews = [['all', 'All stages'], ['route', 'My route']].map(function (v) { var on = s.mapView === v[0]; return { label: v[1], pressed: String(on), style: segStyle(on), pick: function () { self.setState({ mapView: v[0] }); } }; });
    var mapNote = mapRoute ? 'Only the modules on your route are listed; each phase can show its alternatives.' : 'Phases with a dark number are on your route.';

    var selVals = { stage: '', label: '', what: '', inputs: [], outputs: [], fromEarlier: false, hasOut: false, outNote: '', outHead: [], outRows: [], doc: '', status: '', statusStyle: '', canTry: false, tryLabel: '', tryIt: function () {} };
    if (s.selMod && LABEL[s.selMod]) {
      var m = s.selMod, keys = Object.keys(D.cmd).filter(function (k) { return k.indexOf(m + '|') === 0; });
      var ins = [], outs = [];
      keys.forEach(function (k) { var io = D.io[k] || {}; (io.i || []).forEach(function (p) { if (p.indexOf('output/') !== 0 && ins.indexOf(p) < 0) ins.push(p); }); (io.o || []).forEach(function (p) { if (outs.indexOf(p) < 0) outs.push(p); }); });
      var fromSteps = keys.some(function (k) { return ((D.io[k] || {}).i || []).some(function (p) { return p.indexOf('output/') === 0; }); });
      var nt = keys.filter(testedKey).length;
      var exFor = EXAMPLES.filter(function (e) { return buildRoute(studyFor(e.id), {}, s.data).some(function (r) { return r.m === m; }); })[0];
      selVals = Object.assign({ stage: CATALOG[STAGE_OF[m]][0], label: LABEL[m], what: D.intro[m] || DESC[m],
        inputs: ins.slice(0, 5).map(function (p) { return fileItem('m:' + p, p); }), fromEarlier: fromSteps,
        outputs: (outs.length ? outs.slice(0, 4) : ['See the notebook']).map(function (p) { return { v: p }; }), doc: BASE + (DOC[m] || ''),
        status: SOON[m] ? 'Coming soon' : (keys.length ? nt + ' of ' + keys.length + ' workflow' + (keys.length > 1 ? 's' : '') + ' run on example data' : 'Not in the builder yet'),
        statusStyle: SOON[m] || !keys.length ? chip('var(--sk)', 'var(--ink2)') : (nt === keys.length ? chip('var(--okb)', 'var(--okf)') : chip('var(--stb)', 'var(--stf)')),
        canTry: !!exFor && !SOON[m], tryLabel: exFor ? 'Try it in the ' + exFor.label.split(',')[0] + ' example' : '',
        tryIt: function () { if (exFor) self.setRoute({ mode: 'example', src: 'example', example: exFor.id, study: studyFor(exFor.id), methods: {} }, 'Loaded the ' + exFor.label.split(',')[0] + ' example.'); } }, outOf(m));
    }

    // 2. Begin
    var isExample = s.mode === 'example', isOwn = s.mode === 'own';
    var modes = [['example', 'Try an example', 'Learn from a tested analysis that runs on the repository’s example data.', ''], ['own', 'Build my workflow', 'Choose your goals, then give the files your route needs.', '']].map(function (md) {
      var on = s.mode === md[0];
      return { label: md[1], desc: md[2], foot: md[3], checked: String(on),
        style: 'text-align: left; font: inherit; cursor: pointer; padding: 16px 18px; border-radius: 8px; background: ' + (on ? 'var(--acs)' : 'var(--bg)') + '; border: ' + (on ? '2px solid var(--navy)' : '1px solid var(--ln2)') + '; display: block; width: 100%',
        dot: 'flex: 0 0 18px; width: 18px; height: 18px; border-radius: 9px; box-sizing: border-box; border: ' + (on ? '6px solid var(--navy)' : '2px solid var(--ln2)') + '; background: var(--bg)',
        pick: function () { if (md[0] === 'example') self.setRoute({ mode: 'example', src: 'example', study: studyFor(s.example), methods: {} }); else self.setState({ mode: 'own', src: 'own', msg: '' }); } };
    });
    var examples = EXAMPLES.map(function (e) {
      var rt = buildRoute(studyFor(e.id), {}, s.data), ok = routeOk(rt), on = isExample && s.example === e.id;
      return { label: e.label, title: e.title.replace(/^Example: /, '').replace(/^./, function (c) { return c.toUpperCase(); }), checked: on, ariaChecked: String(on), disabled: false, learn: e.learn, result: e.result, route: 'Route: ' + rt.map(function (r) { return LABEL[r.m]; }).join(' → '),
        steps: rt.length + ' steps · ' + rt.filter(function (r) { return MAIN[r.m]; }).map(function (r) { return LABEL[r.m]; }).join(', '),
        notReady: !ok, tag: ok ? 'Runs on example data' : 'Demonstration: not every step runs on the example data yet', tagStyle: ok ? chip('var(--okb)', 'var(--okf)') : chip('var(--stb)', 'var(--stf)'),
        style: 'text-align: left; font: inherit; cursor: pointer; display: block; width: 100%; padding: 12px 14px; border-radius: 8px; background: var(--bg); border: ' + (on ? '2px solid var(--navy)' : '1px solid var(--ln)') + '; box-shadow: 0 1px 2px rgba(25, 53, 74, 0.05)',
        pick: function () { var x = Object.assign({}, s.expStage); x[e.stage] = true; self.setRoute({ example: e.id, study: studyFor(e.id), methods: {}, expStage: x }); } };
    });
    var st = s.study;
    var setStudy = function (patch) { self.setRoute({ study: Object.assign({}, st, patch) }); };
    var radio = function (key, list) { return list.map(function (o) { var on = st[key] === o[0]; return { label: o[1], checked: on, style: pillStyle(on), pick: function () { var p = {}; p[key] = o[0]; setStudy(p); } }; }); };
    var qPheno = radio('ph', [['bulk', 'Gene expression'], ['splice', 'Splicing'], ['meth', 'DNA methylation'], ['other', 'Other molecular phenotype']]);
    var qStart = radio('start', [['raw', 'Raw assay data'], ['matrix', 'Processed phenotype matrix'], ['sumstats', 'QTL summary statistics']]);
    var qGoals = [['discover', 'Discover molecular QTLs'], ['finemap', 'Fine-map QTL signals'], ['gwas', 'Connect to GWAS'], ['mash', 'Compare across contexts'], ['enrich', 'Test enrichment']].map(function (o) { var on = !!st.goals[o[0]]; return { label: o[1], checked: on, style: pillStyle(on), pick: function () { var g = Object.assign({}, st.goals); g[o[0]] = !on; setStudy({ goals: g }); } }; });
    var OPTS = { fmfrom: ['Fine-map from', [['indiv', 'Individual-level genotypes and phenotypes'], ['sumstats', 'QTL summary statistics + LD reference']]], link: ['Connect to GWAS by', [['colocboost', 'Colocalization across many traits (ColocBoost)'], ['enloc', 'Pairwise colocalization (SuSiE-enloc)'], ['twas', 'Gene-level association (TWAS)']]], enrichq: ['Enrichment question', [['eoo', 'Excess of overlap (EOO)'], ['gsea', 'Pathways (GSEA)'], ['gregor', 'Matched SNP sets (GREGOR)'], ['sldsc', 'Heritability (S-LDSC)']]], geno: ['Genotype QC status', [['vcf', 'Raw VCF'], ['plink_raw', 'PLINK, not yet QC’d'], ['plink_qc', 'PLINK, QC’d with PCs']]], cohorts: ['Cohorts', [['single', 'One cohort'], ['multiple', 'Several cohorts to combine']]], contexts: ['Contexts or cell types', [['single', 'One'], ['multiple', 'Several']]], quantile: ['Interaction QTL', [['no', 'No'], ['interaction', 'Genotype × covariate']]] };
    var sel = function (key, why) { var o = OPTS[key]; return { id: 'opt-' + key, label: o[0], why: why || '', value: st[key], opts: o[1].map(function (x) { return { v: x[0], label: x[1] }; }), pick: function (e) { var p = {}; p[key] = e.target.value; setStudy(p); } }; };
    var contextual = [];
    if (st.goals.finemap && st.start !== 'sumstats') contextual.push(sel('fmfrom', 'because you chose fine-mapping'));
    if (st.goals.gwas) contextual.push(sel('link', 'because you chose GWAS'));
    if (st.goals.enrich) contextual.push(sel('enrichq', 'because you chose enrichment'));
    var allOpts = ['geno', 'cohorts', 'contexts', 'quantile'].map(function (k) { return sel(k); });
    var nonDefault = ['geno', 'cohorts', 'contexts', 'quantile'].filter(function (k) { return st[k] !== STUDY0[k]; }).length;

    // 3. Route
    // the earlier step that makes an input: its declared source if that step is on the route,
    // otherwise the latest earlier step whose declared output contains the path
    var stepOfKey = {}; route.forEach(function (r, i) { r.wfs.forEach(function (w) { stepOfKey[r.m + '|' + w] = i; }); });
    var producer = function (p, i, k) {
      var sk = ((D.src || {})[k] || {})[p];
      if (sk && stepOfKey[sk] !== undefined && stepOfKey[sk] < i) return stepOfKey[sk];
      var best = -1, len = -1;
      route.forEach(function (r, j) { if (j >= i) return; r.wfs.forEach(function (w) { ((D.io[r.m + '|' + w] || {}).o || []).forEach(function (o) { if ((p === o || p.indexOf(o + '/') === 0) && (o.length > len || (o.length === len && j > best))) { best = j; len = o.length; } }); }); });
      return best;
    };
    var own = s.src === 'own';
    var userFiles = {}, userOrder = [];
    var placeholder = function (p) { return '<your_' + nice(p.replace(/^<|>$/g, '').replace(/^path\/to\//, '')).replace(/ /g, '_') + '>'; };
    var routeVals = route.map(function (r, i) {
      var keys = r.wfs.map(function (w) { return r.m + '|' + w; });
      var tested = !r.soon && keys.every(testedKey);
      var open = s.stepOpen === r.id, reads = {};
      var ins = [], files = [];
      keys.forEach(function (k) { ((D.io[k] || {}).i || []).forEach(function (p) {
        if (ins.some(function (x) { return x.raw === p; })) return;
        var pi = p.indexOf('output/') === 0 ? producer(p, i, k) : -1;
        if (pi >= 0) reads[pi + 1] = 1;
        // a file the user supplies is one the module page declares with no producing step (D.ext)
        var user = (D.ext[k] || []).indexOf(p) >= 0, ref = user && REF_RX.test(p);
        if (user) { if (!userFiles[p]) { userFiles[p] = []; userOrder.push(p); } if (userFiles[p].indexOf(i + 1) < 0) userFiles[p].push(i + 1); }
        var src = pi >= 0 ? 'From step ' + (pi + 1) : (p.indexOf('output/') === 0 ? 'Made by a module not on this route' : (user ? (ref ? 'Reference data' : (own ? 'Your file' : 'Example file')) : 'Written by this step’s setup lines'));
        var shown = user && own ? (s.paths[p] || placeholder(p)) : p;
        ins.push(fileItem('r' + i + ':' + p, p, { raw: p, source: src, path: shown, showPath: user && own, name: optLabel(((D.opt || {})[k] || {})[p], p).replace(/ \([^()]*\)$/, ''), hasPreview: !!(D.prev || {})[p] && !(user && own && s.paths[p]) }));
        var first = user ? userFiles[p][0] === i + 1 : false, label = optLabel(((D.opt || {})[k] || {})[p], p);
        files.push({ label: label, source: user && own && !first ? '· same file as step ' + userFiles[p][0] : '· ' + src.charAt(0).toLowerCase() + src.slice(1),
          ask: user && own && first, id: 'path-' + userOrder.indexOf(p), value: s.paths[p] || '', aria: label + ' (your file)', placeholder: '/path/to/your/' + p.replace(/^<|>$/g, '').split('/').pop(),
          set: function (e) { var q = Object.assign({}, s.paths); q[p] = e.target.value; self.setState({ paths: q }); } });
      }); });
      var stepsRead = Object.keys(reads).map(Number).sort(function (a, b) { return a - b; });
      var outs = []; keys.forEach(function (k) { ((D.io[k] || {}).o || []).forEach(function (o) { if (outs.indexOf(o) < 0) outs.push(o); }); });
      var choice = r.choices ? r.choices.filter(function (c) { return c[0] === r.chosen; })[0] : null;
      var why = (r.auto ? 'Added to the route because a later step reads the files it makes. ' : '') + (D.intro[r.m] || DESC[r.m] || '');
      if (stepsRead.length) why += ' It reads the output of step ' + stepsRead.join(' and ') + '.';
      return Object.assign({
        n: i + 1, anchor: 'step-' + r.id, title: r.title, module: LABEL[r.m] || r.m, changed: !!s.changed[r.id],
        methodLine: r.soon ? 'coming soon' : (choice ? choice[1] : (r.wfs.length > 1 ? r.wfs.length + ' workflows' : r.wfs[0])),
        metaLine: (LABEL[r.m] || r.m) + ' · ' + (r.soon ? 'coming soon' : (choice ? choice[1] : (r.wfs.length > 1 ? r.wfs.length + ' workflows' : r.wfs[0]))) + (stepsRead.length ? ' · uses step ' + stepsRead.join(', ') : ''),
        cardStyle: 'border: 1px solid ' + (s.changed[r.id] ? 'var(--stf)' : 'var(--ln)') + '; border-radius: 8px; padding: 12px 14px; background: var(--bg)',
        numStyle: 'flex: 0 0 28px; width: 28px; height: 28px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; background: ' + (r.soon ? 'var(--sk); color: var(--ink2)' : 'var(--navy); color: var(--on)'),
        notTested: r.soon || !tested, tested: r.soon ? 'Coming soon' : (tested ? 'Runs on example data' : 'Not yet run on example data'), testStyle: tested ? chip('var(--okb)', 'var(--okf)') : chip('var(--stb)', 'var(--stf)'),
        hasTime: !!D.bench[r.m], time: D.bench[r.m] ? 'Example: ' + D.bench[r.m] : '',
        // step details open as a dialog, one at a time; closing returns focus to the step's button
        isOpen: open, btnId: 'xqb-stepbtn-' + r.id, btn: 'Details',
        btnStyle: 'flex: 0 0 auto; font-size: 15px; min-height: 36px; padding: 6px 14px; border-radius: 6px; cursor: pointer; border: 1px solid var(--ln2); background: var(--bg); color: var(--navy); font-weight: 500',
        toggle: function () { app.focusNext = 'xqb-step-h'; self.setState({ stepOpen: r.id, phase: null, stream: null, selMod: '' }); },
        close: function () { app.focusNext = 'xqb-stepbtn-' + r.id; self.setState({ stepOpen: null }); },
        backdrop: function (e) { if (e.target === e.currentTarget) { app.focusNext = 'xqb-stepbtn-' + r.id; self.setState({ stepOpen: null }); } },
        why: why, hasChoices: !!r.choices, radioName: 'm-' + r.id,
        choices: (r.choices || []).map(function (c) { var k = r.m + '|' + c[0], t = testedKey(k); return { label: c[1], desc: c[3], checked: c[0] === r.chosen, tag: !D.cmd[k] ? 'No command' : (t ? 'Runs on example data' : 'Not yet run'), tagStyle: t ? chip('var(--okb)', 'var(--okf)', ' margin-left: 4px;') : chip('var(--stb)', 'var(--stf)', ' margin-left: 4px;'), pick: function () { var mm = Object.assign({}, s.methods); mm[r.id] = c[0]; self.setRoute({ methods: mm }, c[1] + ' selected.'); } }; }),
        inputs: ins, files: files, hasFiles: own && files.length > 0, outputs: outList(r.m, outs), doc: BASE + (DOC[r.m] || '')
      }, outOf(r.m));
    });
    var nTested = routeVals.filter(function (r) { return r.tested === 'Runs on example data'; }).length;

    // 4. Commands
    var stepMod = function (n) { return route[n - 1] ? route[n - 1].m : ''; };
    var pathFields = userOrder.map(function (p, idx) {
      var pv = (D.prev || {})[p], m = stepMod(userFiles[p][0]), ext = (p.match(/\.([a-z0-9]+)(\.gz)?$/i) || ['', 'file'])[1].toUpperCase();
      var fmtTxt = pv ? ext + ' table, ' + pv.cols + ' columns, first column ' + pv.rows[0][0] : ext + ' file';
      return fileItem('p:' + p, p, { id: 'path-' + idx, steps: userFiles[p].join(', '), format: fmtTxt, module: LABEL[m] || m, doc: BASE + (DOC[m] || ''), ref: REF_RX.test(p),
        btn: s.pv['p:' + p] ? 'Hide the example file' : 'See the expected format (example file)', value: s.paths[p] || '', placeholder: '/path/to/your/' + p.split('/').pop(),
        set: function (e) { var q = Object.assign({}, s.paths); q[p] = e.target.value; self.setState({ paths: q }); } });
    });
    var pathGroups = [['YOUR STUDY FILES', false], ['REFERENCE RESOURCES', true]].map(function (g) { return { label: g[0], fields: pathFields.filter(function (f) { return f.ref === g[1]; }) }; }).filter(function (g) { return g.fields.length; });
    var missing = userOrder.filter(function (p) { return !s.paths[p]; }).length;
    var soonSteps = route.map(function (r, i) { return r.soon ? i + 1 : 0; }).filter(Boolean);
    var cluster = s.runOn === 'local' ? '' : ' \\\n    -c my_cluster.yml -q ' + s.runOn;
    var noCmd = [], exampleOnly = 0;
    var blocks = route.map(function (r, i) {
      var head = '# Step ' + (i + 1) + '. ' + r.title + ' (' + (LABEL[r.m] || r.m) + ')';
      if (r.soon) return head + '\necho "Step ' + (i + 1) + ' has no command yet: see its notebook." >&2; exit 1';
      return head + '\n' + r.wfs.map(function (w) {
        var k = r.m + '|' + w, c = D.cmd[k];
        if (!c) { noCmd.push(i + 1); return 'echo "Step ' + (i + 1) + ': the builder has no command for ' + w + '; see its notebook." >&2; exit 1'; }
        c = fmt(c, cluster);
        if (own) {
          // swap exactly the inputs the module page declares as user-supplied
          (D.ext[k] || []).forEach(function (p) { c = swapPath(c, p, s.paths[p] || placeholder(p)); });
          // anything else still pointing at example files (setup lines that copy example data) must not run silently
          c = c.split('\n').map(function (l) { if (l.indexOf('tests/fixtures/') < 0) return l; exampleOnly++; return '# EXAMPLE DATA, replace with your own: ' + l; }).join('\n');
        }
        return c;
      }).join('\n\n');
    });
    var body = blocks.join('\n\n');
    routeVals.forEach(function (rv, i) { rv.cmd = blocks[i]; var tl = (D.terms || {})[route[i].m] || []; rv.terms = tl.map(function (t) { return { k: t[0], v: t[1] }; }); rv.hasTerms = tl.length > 0; });
    var holes = (body.match(/<your_[^>]+>/g) || []).filter(function (h, n, a) { return a.indexOf(h) === n; }).length;
    var untestedSteps = route.map(function (r, i) { return !r.soon && !r.wfs.every(function (w) { return testedKey(r.m + '|' + w); }) ? i + 1 : 0; }).filter(Boolean);
    var blockers = soonSteps.concat(noCmd);
    var plural = function (n, w) { return n + ' ' + w + (n === 1 ? '' : 's'); };
    // Runnable or template: said in the script's header and in a box above it
    var status = '', ready = false;
    if (route.length) {
      if (blockers.length) status = 'Step ' + blockers.join(', ') + ' has no command in the builder, so the script stops there.';
      else if (own && (holes || exampleOnly)) status = [holes ? 'Fill in ' + plural(holes, 'input path') + ' above' : '', exampleOnly ? 'replace ' + plural(exampleOnly, 'line') + ' marked EXAMPLE DATA with your own files' : ''].filter(Boolean).join(', and ').replace(/^./, function (c) { return c.toUpperCase(); }) + '.';
      else if (own) { ready = true; status = 'Your paths are filled in. Each step has been run on the example data' + (untestedSteps.length ? ' except step ' + untestedSteps.join(', ') : '') + ', not on your data, so check the first steps’ output.'; }
      else if (untestedSteps.length) status = 'Step ' + untestedSteps.join(', ') + ' has not been run on the example data, so this is a demonstration, not a tested script.';
      else { ready = true; status = 'Every step of this route runs on the example data as written.'; }
    }
    var refFiles = userOrder.filter(function (p) { return REF_RX.test(p); }).length;
    var readiness = !route.length ? [] : [
      blockers.length ? { k: 'Route', v: 'step ' + blockers.join(', ') + ' has no command in the builder', mark: '!', cls: 'is-todo' } : { k: 'Route resolved', v: 'every input has a producing step or a file', mark: '✓', cls: 'is-ok' },
      untestedSteps.length ? { k: 'Commands tested', v: (route.length - untestedSteps.length) + ' of ' + route.length + ' steps run on example data', mark: '!', cls: 'is-todo' } : { k: 'Commands tested', v: 'all ' + route.length + ' steps run on example data', mark: '✓', cls: 'is-ok' },
      !own ? { k: 'Input files', v: 'provided by the example', mark: '✓', cls: 'is-ok' } : (missing ? { k: 'Your files', v: missing + ' of ' + userOrder.length + ' paths still needed', mark: '!', cls: 'is-todo' } : { k: 'Your files', v: 'all ' + userOrder.length + ' paths filled in; this page cannot check that they exist', mark: '✓', cls: 'is-ok' }),
      { k: 'Reference data', v: refFiles ? refFiles + ' reference file' + (refFiles === 1 ? '' : 's') + (own ? ' to point to' : ' from the example') : 'none needed for this route', mark: refFiles && own ? '!' : '✓', cls: refFiles && own ? 'is-todo' : 'is-ok' },
      s.runOn === 'local' ? { k: 'Cluster settings', v: 'not needed for local execution', mark: '✓', cls: 'is-ok' } : { k: 'Cluster settings', v: 'check account and queue in my_cluster.yml; this page cannot', mark: '!', cls: 'is-todo' }
    ];
    readiness = readiness.filter(function (c) { return c.cls !== 'is-ok'; });
    var script = route.length ? '#!/bin/bash\n# xQTL analysis route from the Workflow Builder\n# Protocol revision: ' + REV + '\n# ' + (own ? 'Your data' : 'Repository example data') + ' · ' + { local: 'runs here', slurm: 'SoS submits each step to SLURM', sge: 'SoS submits each step to SGE' }[s.runOn] + '\n# Status: ' + (ready ? 'ready to run. ' : 'NOT RUNNABLE YET. ') + status + '\nset -euo pipefail\n\n' + body : '# Choose an example or describe your study to build the route.';
    var setupOnce = [{ v: 'Clone the repository.' }, { v: 'Install the software environment with pixi (see Environment setup).' }];
    if (s.runOn !== 'local') setupOnce.push({ v: 'Copy code/cluster/' + s.runOn + '.yml to my_cluster.yml and set your account and queue.' });
    var setupRoute = [{ v: 'Open a terminal in the repository folder' + (s.runOn === 'local' ? '.' : ', on the cluster’s login node.') }];
    if (own) setupRoute.push({ v: missing ? 'Fill in the ' + missing + ' input path' + (missing > 1 ? 's' : '') + ' above; the script shows <your_…> until you do.' : 'Check that each path above points at a readable file.' });
    else setupRoute.push({ v: 'The example files are already in the repository; nothing to download.' });
    setupRoute.push({ v: s.runOn === 'local' ? 'Run the script: every step runs on this computer, one after another.' : 'Run the script: SoS sends each step to ' + (s.runOn === 'slurm' ? 'SLURM' : 'SGE') + ' and waits for it. This page does not submit anything itself.' });
    var mainResults = [], inter = [];
    route.forEach(function (r, i) { r.wfs.forEach(function (w) { ((D.io[r.m + '|' + w] || {}).o || []).forEach(function (o) {
      if (MAIN[r.m] && !mainResults.some(function (x) { return x.m === r.m; })) mainResults.push(Object.assign({ m: r.m, n: i + 1, module: LABEL[r.m], title: MAIN[r.m][0], meaning: MAIN[r.m][1], next: MAIN[r.m][2], path: o }, outOf(r.m)));
      else if (!inter.some(function (e) { return e.path === o; }) && !mainResults.some(function (x) { return x.path === o; })) inter.push({ n: i + 1, path: o });
    }); }); });
    var loadExample = function () { self.setRoute({ mode: 'example', src: 'example', study: studyFor(s.example), methods: {} }, 'Loaded the example route.'); scrollTo('r-h'); };
    var resetAll = function () { self.setRoute({ mode: 'example', src: 'example', example: 'eqtl', study: studyFor('eqtl'), methods: {}, paths: {}, runOn: 'local', phase: null, stream: null, selMod: '' }, 'Reset to the eQTL example.'); };
    var copyLink = function () { var u = shareUrl(s); copyToClipboard(u, function (ok) { self.setState({ copied: ok ? 'link' : 'fail' }); setTimeout(function () { self.setState({ copied: '' }); }, 2000); }); };
    var exObj = EXAMPLES.filter(function (e) { return e.id === s.example; })[0] || EXAMPLES[0];
    var copyText = function (text, which) { copyToClipboard(text, function (ok) { self.setState({ copied: ok ? which : 'fail' }); setTimeout(function () { self.setState({ copied: '' }); }, 2000); }); };

    // 5. Catalog
    var qq = (s.q || '').toLowerCase();
    var catalog = CATALOG.map(function (g) {
      var items = g[2].map(function (m) {
        var ks = Object.keys(D.cmd).filter(function (k) { return k.indexOf(m[0] + '|') === 0; }), nt = ks.filter(testedKey).length, soon = !!SOON[m[0]];
        return { id: m[0], label: m[1], desc: m[2], doc: BASE + (DOC[m[0]] || ''), onRoute: !!onRoute[m[0]], soon: soon, notReady: soon || !ks.length || nt < ks.length,
          status: soon ? 'Coming soon' : (ks.length ? nt + ' of ' + ks.length + ' run on example data' : 'See notebook'),
          statusStyle: soon || !ks.length ? chip('var(--sk)', 'var(--ink2)') : (nt === ks.length ? chip('var(--okb)', 'var(--okf)') : chip('var(--stb)', 'var(--stf)')) };
      }).filter(function (it) { if (s.catOnly && s.catOnly.indexOf(it.id) < 0) return false; if (qq && (it.label + ' ' + it.desc + ' ' + it.id).toLowerCase().indexOf(qq) < 0) return false; if (s.filter === 'route') return it.onRoute; if (s.filter === 'alt') return !it.onRoute && !it.notReady; if (s.filter === 'notready') return it.notReady; return true; });
      return { label: g[0].toUpperCase(), items: items };
    }).filter(function (g) { return g.items.length; });
    var filters = [['all', 'All modules'], ['route', 'On my route'], ['alt', 'Alternatives'], ['notready', 'Not ready']].map(function (f) { var on = s.filter === f[0]; return { label: f[1], pressed: String(on), pick: function () { self.setState({ filter: f[0] }); }, style: 'font-size: 14px; min-height: 34px; padding: 5px 12px; border-radius: 6px; cursor: pointer; border: 1px solid ' + (on ? 'var(--navy)' : 'var(--ln2)') + '; background: ' + (on ? 'var(--navy)' : 'var(--bg)') + '; color: ' + (on ? 'var(--bg)' : 'var(--ink)') }; });

    // the eight-phase map (explains the science; the route above says what runs)
    var scrollTo = function (id) { setTimeout(function () { var t = R.getElementById(id); if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 60); };
    var modStatus = function (id) {
      var ks = Object.keys(D.cmd).filter(function (k) { return k.indexOf(id + '|') === 0; }), nt = ks.filter(testedKey).length, soon = !!SOON[id];
      return { ks: ks, soon: soon, notReady: soon || !ks.length || nt < ks.length, status: soon ? 'Coming soon' : (ks.length ? nt + ' of ' + ks.length + ' workflow' + (ks.length === 1 ? '' : 's') + ' run on example data' : 'No builder command; see notebook'),
        statusStyle: soon || !ks.length ? chip('var(--sk)', 'var(--ink2)') : (nt === ks.length ? chip('var(--okb)', 'var(--okf)') : chip('var(--stb)', 'var(--stf)')) };
    };
    var modsOf = function (pi, gi) { var out = []; FLOW[pi][2].forEach(function (bx) { bx[1].forEach(function (gr, g) { if (gi == null || g === gi) out = out.concat(gr[1]); }); }); return out; };
    // the flowchart: each phase lists its modules in boxes and groups (FLOW), as on the original builder
    var flow = FLOW.map(function (ph, pi) {
      return { num: String(pi + 1), kicker: ph[0], hasArrow: pi > 0, rowCls: 'xqb-frow' + (ph[2].length > 1 ? ' is-multi' : ''),
        boxes: ph[2].map(function (bx, bi) {
          var mods = []; bx[1].forEach(function (gr) { mods = mods.concat(gr[1]); });
          var on = mods.some(function (id) { return onRoute[id]; }), bid = 'xqb-node-' + pi + '-' + bi;
          return { title: bx[0], sub: bx[1].filter(function (gr) { return gr[0]; }).map(function (gr) { return gr[0]; }).join(' · '),
            hasSub: bx[1].some(function (gr) { return gr[0]; }), bid: bid, cls: 'xqb-fbox' + (on ? ' is-route' : ''),
            aria: bx[0] + (on ? ', on your route' : '') + ', show its modules',
            pick: function () { app.focusNext = 'xqb-phase-h'; self.setState({ phase: pi, stream: null, selMod: '', phaseFrom: bid }); } }; }) };
    });
    var phaseVals = null, modVals = {};
    // the phase or module details open as a dialog; closing returns focus to what opened it
    var closePanel = function () { app.focusNext = s.phaseFrom || ('xqb-node-' + s.phase + '-0'); self.setState({ phase: null, stream: null, selMod: '' }); };
    if (s.phase != null && FLOW[s.phase]) {
      var pi = s.phase, gi = s.stream, ph = FLOW[pi], preset = PHASES[pi][2];
      var groups = [];
      ph[2].forEach(function (bx) { bx[1].forEach(function (gr, g) {
        groups.push({ title: gr[0] || (ph[2].length > 1 ? bx[0] : ''), cls: 'xqb-g' + (gi != null && g !== gi ? ' is-faint' : ''), mods: gr[1].map(function (id) {
          var on = !!onRoute[id], st = modStatus(id), sel = s.selMod === id;
          return { label: LABEL[id] || id, pressed: String(sel), state: on ? ' · on your route' : (st.soon ? ' · coming soon' : (st.notReady ? ' · not fully tested' : '')),
            mid: 'xqb-mod-' + String(id).replace(/[^A-Za-z0-9]+/g, '-'), soon: st.soon, cls: 'xqb-fm' + (on ? ' is-route' : '') + (st.soon ? ' is-soon' : ''),
            pick: function () { var step = stepOf[id]; if (step) { app.focusNext = 'xqb-step-h'; self.setState({ stepOpen: step, phase: null, stream: null, selMod: '' }); return; } app.focusNext = 'xqb-phase-h'; self.setState({ selMod: id }); } };
        }) });
      }); });
      if (s.selMod && LABEL[s.selMod]) {
        var m = s.selMod, st = modStatus(m), ins = [], seen = {}, outs = [];
        // inputs one per line, by name, with the example file's preview where there is one
        st.ks.forEach(function (k) { var io = D.io[k] || {}; (io.i || []).forEach(function (p) {
          var name = optLabel(((D.opt || {})[k] || {})[p], p).replace(/ \([^()]*\)$/, '');
          if (seen[name]) return; seen[name] = 1;
          ins.push(fileItem('m:' + m + ':' + p, p, { name: name, n: ins.length + 1 }));
        }); (io.o || []).forEach(function (o) { if (outs.indexOf(o) < 0) outs.push(o); }); });
        var outItems = outList(m, outs);
        var gIdx = -1; ph[2].forEach(function (bx) { bx[1].forEach(function (gr, g) { if (gr[1].indexOf(m) >= 0) gIdx = g; }); });
        modVals = Object.assign({ label: LABEL[m], what: D.intro[m] || DESC[m], doc: BASE + (DOC[m] || ''),
          when: pi === 2 && gIdx >= 0 ? STREAMS[gIdx][2] : (PHASES[pi][1] || DESC[m]),
          inputs: ins, hasInputs: ins.length > 0, outputs: outItems, hasOutputs: outs.length > 0 });
      }
      phaseVals = { kicker: 'PHASE ' + (pi + 1) + (gi != null ? ' · ' + STREAMS[gi][0].toUpperCase() + ' STREAM' : ''),
        label: ph[0] === 'Inputs' ? 'Reference data preparation' : (pi === 2 ? 'Data pre-processing' : (pi === 3 ? 'QTL association testing' : (pi === 5 ? 'Fine-mapping' : (pi === 6 ? 'GWAS integration' : (pi === 7 ? 'Enrichment, validation and interpretation' : (pi === 1 ? 'Molecular phenotype quantification' : 'Multivariate modeling')))))),
        what: pi === 2 ? (gi != null ? STREAMS[gi][2] : 'Genotype, phenotype and covariate preparation run side by side and converge on association testing.') : PHASES[pi][0] + '. ' + PHASES[pi][1],
        srHint: (gi != null ? STREAMS[gi][0] + ' stream, ' : '') + groups.reduce(function (a, g) { return a + g.mods.length; }, 0) + ' modules listed below. Close returns you to the map',
        groups: groups, hasMod: !!modVals.label, showList: !modVals.label, back: function () { app.focusNext = 'xqb-phase-h'; self.setState({ selMod: '' }); }, canBuild: !!preset, buildLabel: preset ? preset.label : '',
        build: function () {
          var ex = preset.example && s.mode === 'example' ? preset.example : '';
          if (ex) self.setRoute({ example: ex, study: studyFor(ex), methods: {} }, 'Loaded the matching example.');
          else { var st0 = clone(STUDY0); Object.keys(preset.study).forEach(function (k2) { st0[k2] = k2 === 'goals' ? Object.assign(clone(STUDY0.goals), preset.study.goals) : preset.study[k2]; });
            self.setRoute({ mode: 'own', src: 'own', study: st0, methods: {} }, 'Applied the study preset for this phase.'); }
          scrollTo('r-h');
        },
        close: closePanel,
        backdrop: function (e) { if (e.target === e.currentTarget) closePanel(); } };
    }

    var testedCount = Object.keys(D.cmd).filter(testedKey).length;
    var journey = [['Understand the workflow', 'Stages and what each module does', '#map-h'], ['Choose how to begin', 'Example data or your own', '#begin-h'], ['Review your route', route.length ? route.length + ' steps, in order' : 'Modules in order', '#r-h'], ['Generate commands', 'Local, SLURM or SGE', '#c-h']].map(function (j, i) {
      return { n: i + 1, label: j[0], sub: j[1], href: j[2], style: 'padding: 12px 14px; background: ' + (i === 0 ? 'var(--tint)' : 'var(--bg)') + '; border-right: 1px solid var(--ln)' };
    });

    return {
      testedCount: testedCount, journey: journey, flow: flow, hasPhase: !!phaseVals, phase: phaseVals || {}, mod: modVals, setupDoc: BASE + 'xqtl-protocol-demo',
      readiness: readiness, loadExample: loadExample, resetAll: resetAll, copyLink: copyLink, copyLinkLabel: s.copied === 'link' ? 'Link copied' : 'Copy a link to this setup',
      catOpen: !!s.catOpen, catExpanded: String(!!s.catOpen), catBtn: s.catOpen ? 'Hide the module catalog' : 'Show all ' + Object.keys(LABEL).length + ' modules', toggleCat: function () { self.setState({ catOpen: !s.catOpen, catOnly: null }); },
      hasCatOnly: !!s.catOnly, catOnlyLabel: s.catOnly ? 'Showing the ' + (s.catOnlyLabel || '') + ' modules.' : '', clearCatOnly: function () { self.setState({ catOnly: null }); }, moduleCount: Object.keys(LABEL).length, clusterDoc: BASE + 'running-on-a-cluster',
      stageNav: CATALOG.map(function (g) { return { label: g[0] }; }),
      mapViews: mapViews, mapNote: mapNote, map: map, hasSel: !!selVals.label, sel: selVals, closeSel: function () { self.setState({ selMod: '' }); },
      modes: modes, isExample: isExample, isOwn: isOwn, examples: examples,
      qPheno: qPheno, qStart: qStart, qGoals: qGoals, hasContextual: contextual.length > 0, contextual: contextual,
      adv: s.adv, advExpanded: String(s.adv), toggleAdv: function () { self.setState({ adv: !s.adv }); }, advLabel: s.adv ? 'Hide advanced study settings' : 'Advanced study settings', advCount: nonDefault ? nonDefault + ' changed' : 'genotype QC, cohorts, contexts', allOpts: allOpts,
      msg: s.msg, msgStyle: s.msg ? 'margin: 12px 0 0; padding: 8px 12px; border-radius: 6px; background: var(--tint); border-left: 3px solid var(--teal); color: var(--ink); font-size: 15px' : 'margin: 0; height: 0; overflow: hidden',
      route: routeVals, routeEmpty: route.length === 0, 
      srcTabs: [['example', 'Example data'], ['own', 'My own data']].map(function (t) { var on = s.src === t[0]; return { label: t[1], pressed: String(on), style: segStyle(on), pick: function () { self.setState({ src: t[0], mode: t[0] }); } }; }),
      runOn: [['local', 'Local terminal'], ['slurm', 'SLURM'], ['sge', 'SGE']].map(function (p) { var on = s.runOn === p[0]; return { label: p[1], checked: on, style: pillStyle(on), pick: function () { self.setState({ runOn: p[0] }); } }; }),
      showCluster: s.runOn !== 'local', runOnName: s.runOn === 'slurm' ? 'SLURM' : 'SGE', runOnFile: s.runOn, nativeCmd: s.runOn === 'slurm' ? 'sbatch' : 'qsub',
      isOwnSrc: own, isExampleSrc: !own && route.length > 0, pathGroups: pathGroups, noPaths: own && pathFields.length === 0,
      pathStatus: missing ? missing + ' of ' + pathFields.length + ' still needed' : 'Complete', pathStatusStyle: missing ? chip('var(--stb)', 'var(--stf)') : chip('var(--okb)', 'var(--okf)'),
      setupOnce: setupOnce, setupRoute: setupRoute, incomplete: soonSteps.length > 0, incompleteText: 'Step ' + soonSteps.join(', ') + ' has no command in the builder yet, so the script stops before it. Choose another method for that step or run it from its notebook.',
      mainResults: mainResults, noResults: route.length === 0, inter: inter, hasInter: inter.length > 0, interCount: String(inter.length),
      hasProduces: mainResults.length > 0, produces: 'This analysis produces ' + (function (t) { t = t.map(function (x) { return x.charAt(0).toLowerCase() + x.slice(1); }); return t.length > 1 ? t.slice(0, -1).join(', ') + ' and ' + t[t.length - 1] : t[0]; })(mainResults.map(function (m) { return m.title; })) + '.',
      jumpOwn: function () { self.setState({ mode: 'own', src: 'own', msg: '' }); },
      tut: { title: exObj.title, items: [{ k: 'WHAT YOU LEARN', v: exObj.learn }, { k: 'PROVIDED INPUTS', v: exObj.given }, { k: 'EXPECTED RESULT', v: exObj.result }, { k: 'AFTERWARDS', v: exObj.after }] },
      script: script, hasStatus: !!status, status: status, statusStyle: 'border-radius: 6px; padding: 10px 14px; font-size: 15px; line-height: 1.5; margin-bottom: 10px; background: var(--tint); color: var(--ink); border-left: 3px solid ' + (ready ? 'var(--okf)' : 'var(--stf)'), statusHead: ready ? 'Ready to run.' : 'Not a runnable script yet.', fileName: own ? 'xqtl_route.sh' : 'xqtl_route_example.sh', srcLabel: own ? 'your data' : 'example data',
      copyScriptLabel: s.copied === 'script' ? 'Copied' : (s.copied === 'fail' ? 'Copy failed' : 'Copy all'), copyScript: function () { copyText(script, 'script'); },
      downloadScript: function () { try { var b = new Blob([script + '\n'], { type: 'text/x-shellscript' }); var a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = own ? 'xqtl_route.sh' : 'xqtl_route_example.sh'; a.click(); setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000); } catch (e) {} },
      ownNeed: String(userOrder.length), useWithMine: function () { self.setState({ mode: 'own', src: 'own', msg: 'Kept your route and methods. Enter your files in the route steps.' }); scrollTo('r-h'); },
      q: s.q, onSearch: function (e) { self.setState({ q: e.target.value }); }, filters: filters, catalog: catalog, catalogEmpty: catalog.length === 0
    };
  }
}

/* ---- Mount ---- */
// the template is XML so that <sc-for> stays inside table rows (an HTML parser would move it out)
var root = R.getElementById('xqb');
var tpl = new DOMParser().parseFromString(R.getElementById('xqb-tpl').textContent, 'application/xml').documentElement;
var app = new Component(), pending = false;
app.setState = function (patch) { Object.assign(app.state, patch); if (!pending) { pending = true; (window.requestAnimationFrame || setTimeout)(function () { pending = false; paint(); }); } };
function paint() {
  // keep focus, caret and open <details> across re-renders
  var a = R.activeElement, fid = a && root.contains(a) && a.id, sel = null;
  if (fid && typeof a.selectionStart === 'number') sel = [a.selectionStart, a.selectionEnd];
  var open = {}; Array.prototype.forEach.call(root.querySelectorAll('details[id]'), function (d) { open[d.id] = d.open; });
  var frag = document.createDocumentFragment();
  kids(tpl, [app.renderVals()], frag);
  root.replaceChildren(frag);
  Object.keys(open).forEach(function (id) { var d = R.getElementById(id); if (d) d.open = open[id]; });
  var jump = !!app.focusNext; if (jump) { fid = app.focusNext; sel = null; app.focusNext = null; }
  if (fid) { var f = R.getElementById(fid); if (f) { f.focus(jump ? { preventScroll: true } : undefined); if (sel && f.setSelectionRange) try { f.setSelectionRange(sel[0], sel[1]); } catch (e) {} } }
}
// in-page links scroll here instead of sending the parent page to the builder's own URL
// dialogs: Esc closes, Tab stays inside, focus returns to what opened it
root.addEventListener('keydown', function (e) {
  var st = app.state, dlg = R.querySelector('[role=dialog]');
  if (!dlg) return;
  if (e.key === 'Tab') {
    var f = Array.prototype.filter.call(dlg.querySelectorAll('a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])'), function (x) { return !x.disabled && x.offsetParent !== null; });
    if (!f.length) return;
    var first = f[0], last = f[f.length - 1], a = R.activeElement;
    if (e.shiftKey && (a === first || !dlg.contains(a))) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && (a === last || !dlg.contains(a))) { e.preventDefault(); first.focus(); }
    return;
  }
  if (e.key !== 'Escape') return;
  if (st.stepOpen) { app.focusNext = 'xqb-stepbtn-' + st.stepOpen; app.setState({ stepOpen: null }); return; }
  if (st.phase == null) return;
  app.focusNext = st.phaseFrom || ('xqb-node-' + st.phase + '-0');
  app.setState({ phase: null, stream: null, selMod: '' });
});
root.addEventListener('click', function (e) {
  var a = e.target.closest && e.target.closest('a[href^="#"]'); if (!a) return;
  var t = R.getElementById(a.getAttribute('href').slice(1)); e.preventDefault();
  if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
});
var shared = readShared(); if (shared) Object.assign(app.state, shared);
// dark theme: follow the site's toggle (on the site's page or in its frame), else the system setting
var host = WIDGET ? document : (function () { try { return window.self !== window.top ? window.parent.document : null; } catch (e) { return null; } })();
var themeEl = WIDGET ? root.parentNode : document.documentElement;
function applyTheme() {
  var mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null, dark;
  if (host) { var h = host.documentElement; dark = h.classList.contains('dark') || h.getAttribute('data-theme') === 'dark'; }
  else dark = !!(mq && mq.matches);
  themeEl.classList.toggle('xqb-dark', dark);
}
applyTheme();
try { if (host) new MutationObserver(applyTheme).observe(host.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] }); } catch (e) {}
try { window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme); } catch (e) {}
paint();
})();
