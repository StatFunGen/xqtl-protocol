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
var XQB_DATA = JSON.parse(document.getElementById('xqb-data').textContent);
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
var SOON = {METAL: 1, watershed: 1, ems_prediction: 1, qr_and_twas: 1};
var LABEL = {}, STAGE_OF = {}, DESC = {};
CATALOG.forEach(function (g, gi) { g[2].forEach(function (m) { LABEL[m[0]] = m[1]; STAGE_OF[m[0]] = gi; DESC[m[0]] = m[2]; }); });
var FIXED_CHOICES = {
  TensorQTL: [['cis', 'Cis scan', true, 'Tests variants near each gene. The protocol’s main analysis.'], ['interaction', 'Interaction scan', false, 'Cis scan with a genotype × covariate term, for example sex.'], ['trans', 'Trans scan', false, 'Tests variants far from each gene, on other chromosomes too.']],
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
var REF_RX = /(ld_reference|\/reference\/|\.gtf|\.fa(sta)?(\.gz)?$|ld_meta|LD_blocks)/i;
var EXAMPLES = [
  { id: 'eqtl', stage: 3, label: 'eQTL discovery, starting from prepared data', study: { goals: { discover: true } }, title: 'Example: discover eQTLs', learn: 'How genotypes and gene expression are combined to find variants associated with each gene’s expression.', given: 'Genotypes, an expression matrix and covariates for a small set of samples on chromosome 22.', result: 'Variant–expression association results and the list of genes with a significant eQTL.', after: 'Open the results table, then use the same workflow with your own data.' },
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
    if (g.discover || g.mash || s.quantile === 'interaction') { addChoice('tq', 'TensorQTL', 'Association scan', s.quantile === 'interaction' ? 'interaction' : null); add('pp', 'qtl_association_postprocessing', ['default'], 'Multiple-testing correction'); }
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
    (io.i || []).forEach(function (p) { if (p.indexOf('output/') === 0) { var pk = producer(p); if (pk && pk !== k) ks.push(pk); } });
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


class Component {
  constructor() {
    this.state = { data: XQB_DATA, mode: 'example', example: 'eqtl', study: studyFor('eqtl'), methods: {}, src: 'example', runOn: 'local', paths: {}, mapView: 'all', selMod: '', open: {}, pv: {}, expStage: {}, changed: {}, msg: '', adv: false, q: '', filter: 'all', copied: '' };
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
    var chip = function (bg, fg, extra) { return 'font-size: 12.5px; padding: 2px 8px; border-radius: 4px; background: ' + bg + '; color: ' + fg + ';' + (extra || ''); };
    var tableOf = function (pv) { return { head: pv ? pv.rows[0].map(function (v) { return { v: v }; }) : [], rows: pv ? pv.rows.slice(1).map(function (row) { return { cells: row.map(function (v) { return { v: v }; }) }; }) : [] }; };
    var fileItem = function (key, p, extra) {
      var pv = (D.prev || {})[p], on = !!s.pv[key] && !!pv, t = tableOf(pv);
      return Object.assign({ label: nice(p).replace(/^./, function (c) { return c.toUpperCase(); }), path: p, hasPreview: !!pv, open: on, expanded: String(on), btn: on ? 'Hide example' : 'Show example',
        note: pv ? 'First ' + (pv.rows.length - 1) + ' rows' + (pv.cols > 6 ? ', 6 of ' + pv.cols + ' columns' : '') + ' of the example file.' : '', head: t.head, rows: t.rows,
        toggle: function () { var q = Object.assign({}, s.pv); q[key] = !q[key]; self.setState({ pv: q }); } }, extra || {});
    };
    var outOf = function (m) { var op = (D.outPrev || {})[m], t = tableOf(op); return { hasOut: !!op, outNote: op ? 'Example output, ' + op.file + ': first ' + (op.rows.length - 1) + ' rows' + (op.cols > 6 ? ', 6 of ' + op.cols + ' columns' : '') + '.' : '', outHead: t.head, outRows: t.rows }; };
    var segStyle = function (on) { return 'font-size: 14px; min-height: 36px; padding: 6px 14px; border: 0; cursor: pointer; background: ' + (on ? '#19354a' : '#ffffff') + '; color: ' + (on ? '#ffffff' : '#202b35') + '; font-weight: ' + (on ? '600' : '400'); };
    var pillStyle = function (on) { return 'display: inline-flex; align-items: center; gap: 8px; font-size: 14px; min-height: 38px; padding: 7px 13px; border-radius: 6px; cursor: pointer; box-sizing: border-box; border: 1px solid ' + (on ? '#19354a' : '#c9d3da') + '; background: ' + (on ? '#e8eef5' : '#ffffff') + '; color: #202b35; font-weight: ' + (on ? '600' : '400'); };

    var route = buildRoute(s.study, s.methods, s.data);
    var onRoute = {}; route.forEach(function (r) { onRoute[r.m] = 1; });
    var routeOk = function (rt) { return rt.length > 0 && rt.every(function (r) { return !r.soon && r.wfs.every(function (w) { return testedKey(r.m + '|' + w); }); }); };

    // 1. Map
    var mapRoute = s.mapView === 'route' && route.length > 0;
    var map = CATALOG.map(function (g, gi) {
      var nOn = g[2].filter(function (m) { return onRoute[m[0]]; }).length;
      var active = !mapRoute || nOn > 0;
      var mods = g[2].filter(function (m) { return !mapRoute || onRoute[m[0]]; }).map(function (m) {
        var sel = s.selMod === m[0], on = !!onRoute[m[0]];
        return { label: m[1], pressed: String(sel), pick: function () { var e = Object.assign({}, s.expStage); e[gi] = true; self.setState({ selMod: sel ? '' : m[0], expStage: e }); },
          style: 'font-size: 13.5px; min-height: 34px; padding: 5px 11px; border-radius: 6px; cursor: pointer; border: 1px solid ' + (sel ? '#19354a' : (on ? '#3d6289' : '#dce3e8')) + '; background: ' + (sel ? '#19354a' : (on ? '#e8eef5' : '#ffffff')) + '; color: ' + (sel ? '#ffffff' : '#202b35') + '; font-weight: ' + (on || sel ? '600' : '400') };
      });
      var exp = mapRoute || !!s.expStage[gi];
      return { n: gi + 1, label: g[0], sub: mapRoute ? nOn + ' module' + (nOn === 1 ? '' : 's') + ' on your route' : g[1], mods: mods,
        showMods: exp, expanded: String(exp), btn: mapRoute ? '' : (exp ? 'Hide modules' : 'Show ' + g[2].length + ' modules'),
        toggle: function () { var e = Object.assign({}, s.expStage); e[gi] = !e[gi]; self.setState({ expStage: e }); },
        titleStyle: 'font-size: 16px; font-weight: 600; color: ' + (active ? '#19354a' : '#63717d'),
        badge: 'width: 28px; height: 28px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; ' + (active ? 'background: #19354a; color: #ffffff' : 'background: #eef2f5; color: #63717d'),
        rail: gi < CATALOG.length - 1 ? 'flex: 1 1 auto; width: 2px; background: #dce3e8; margin: 4px 0' : 'flex: 1 1 auto; width: 2px; background: transparent' };
    });
    var mapViews = [['all', 'All stages'], ['route', 'My route']].map(function (v) { var on = s.mapView === v[0]; return { label: v[1], pressed: String(on), style: segStyle(on), pick: function () { self.setState({ mapView: v[0] }); } }; });
    var mapNote = mapRoute ? 'Showing only the stages and modules on your route.' : 'Open a stage to see its modules. Modules on your route are highlighted.';

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
        statusStyle: SOON[m] || !keys.length ? chip('#eef2f5', '#63717d') : (nt === keys.length ? chip('#e3f1e8', '#1f6b3d') : chip('#fbf0d9', '#7a5300')),
        canTry: !!exFor && !SOON[m], tryLabel: exFor ? 'Try it in the ' + exFor.label.split(',')[0] + ' example' : '',
        tryIt: function () { if (exFor) self.setRoute({ mode: 'example', src: 'example', example: exFor.id, study: studyFor(exFor.id), methods: {} }, 'Loaded the ' + exFor.label.split(',')[0] + ' example.'); } }, outOf(m));
    }

    // 2. Begin
    var isExample = s.mode === 'example', isOwn = s.mode === 'own';
    var modes = [['example', 'Try example data', 'Follow a route that runs on the repository’s example files and see what it produces.', 'Recommended for first-time users'], ['own', 'Use my own data', 'Describe your study, then give the files the route needs.', 'For your research data']].map(function (md) {
      var on = s.mode === md[0];
      return { label: md[1], desc: md[2], foot: md[3], checked: String(on),
        style: 'text-align: left; font: inherit; cursor: pointer; padding: 16px 18px; border-radius: 8px; background: ' + (on ? '#e8eef5' : '#ffffff') + '; border: ' + (on ? '2px solid #19354a' : '1px solid #c9d3da') + '; display: block; width: 100%',
        dot: 'flex: 0 0 18px; width: 18px; height: 18px; border-radius: 9px; box-sizing: border-box; border: ' + (on ? '6px solid #19354a' : '2px solid #c9d3da') + '; background: #ffffff',
        pick: function () { if (md[0] === 'example') self.setRoute({ mode: 'example', src: 'example', study: studyFor(s.example), methods: {} }); else self.setState({ mode: 'own', src: 'own', msg: '' }); } };
    });
    var examples = EXAMPLES.map(function (e) {
      var rt = buildRoute(studyFor(e.id), {}, s.data), ok = routeOk(rt), on = isExample && s.example === e.id;
      return { label: e.label, checked: on, disabled: false, learn: e.learn, result: e.result, route: 'Route: ' + rt.map(function (r) { return LABEL[r.m]; }).join(' → '),
        tag: ok ? 'Runs on example data' : 'Demonstration: not every step runs on the example data yet', tagStyle: ok ? chip('#e3f1e8', '#1f6b3d') : chip('#fbf0d9', '#7a5300'),
        style: 'display: flex; gap: 10px; align-items: flex-start; font-size: 14.5px; padding: 8px 10px; border-radius: 6px; cursor: pointer; background: ' + (on ? '#f6f8fa' : 'transparent'),
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
    var outsOf = {};
    route.forEach(function (r, i) { r.wfs.forEach(function (w) { ((D.io[r.m + '|' + w] || {}).o || []).forEach(function (o) { outsOf[o] = i; }); }); });
    var producer = function (p, i) { var best = -1, len = 0; Object.keys(outsOf).forEach(function (o) { if (outsOf[o] < i && (p === o || p.indexOf(o + '/') === 0) && o.length > len) { best = outsOf[o]; len = o.length; } }); return best; };
    var own = s.src === 'own';
    var userFiles = {}, userOrder = [];
    var routeVals = route.map(function (r, i) {
      var keys = r.wfs.map(function (w) { return r.m + '|' + w; });
      var tested = !r.soon && keys.every(testedKey);
      var open = !!s.open[r.id], reads = {};
      var ins = [];
      keys.forEach(function (k) { ((D.io[k] || {}).i || []).forEach(function (p) {
        if (ins.some(function (x) { return x.raw === p; })) return;
        var pi = p.indexOf('output/') === 0 ? producer(p, i) : -1;
        if (pi >= 0) reads[pi + 1] = 1;
        var fixture = p.indexOf('tests/fixtures/') === 0;
        if (fixture) { if (!userFiles[p]) { userFiles[p] = []; userOrder.push(p); } if (userFiles[p].indexOf(i + 1) < 0) userFiles[p].push(i + 1); }
        var src = pi >= 0 ? 'From step ' + (pi + 1) : (p.indexOf('output/') === 0 ? 'Made by a module not on this route' : (fixture ? (own ? 'Your file' : 'Example file') : 'Reference data'));
        var shown = fixture && own ? (s.paths[p] || '<your ' + nice(p) + '>') : p;
        ins.push(fileItem('r' + i + ':' + p, p, { raw: p, source: src, path: shown, hasPreview: !!(D.prev || {})[p] && !(fixture && own && s.paths[p]) }));
      }); });
      var stepsRead = Object.keys(reads).map(Number).sort(function (a, b) { return a - b; });
      var outs = []; keys.forEach(function (k) { ((D.io[k] || {}).o || []).forEach(function (o) { if (outs.indexOf(o) < 0) outs.push(o); }); });
      var choice = r.choices ? r.choices.filter(function (c) { return c[0] === r.chosen; })[0] : null;
      var why = (r.auto ? 'Added to the route because a later step reads the files it makes. ' : '') + (D.intro[r.m] || DESC[r.m] || '');
      if (stepsRead.length) why += ' It reads the output of step ' + stepsRead.join(' and ') + '.';
      return Object.assign({
        n: i + 1, anchor: 'step-' + r.id, title: r.title, module: LABEL[r.m] || r.m, changed: !!s.changed[r.id],
        methodLine: r.soon ? 'coming soon' : (choice ? choice[1] : (r.wfs.length > 1 ? r.wfs.length + ' workflows' : r.wfs[0])),
        readsLine: stepsRead.length ? 'Uses step ' + stepsRead.join(', ') : 'Starts from input files',
        cardStyle: 'border: 1px solid ' + (s.changed[r.id] ? '#e2c27a' : '#dce3e8') + '; border-radius: 8px; padding: 12px 14px; background: #ffffff',
        numStyle: 'flex: 0 0 28px; width: 28px; height: 28px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px; background: ' + (r.soon ? '#eef2f5; color: #63717d' : '#19354a; color: #ffffff'),
        tested: r.soon ? 'Coming soon' : (tested ? 'Runs on example data' : 'Not yet run on example data'), testStyle: tested ? chip('#e3f1e8', '#1f6b3d') : chip('#fbf0d9', '#7a5300'),
        hasTime: !!D.bench[r.m], time: D.bench[r.m] ? 'Example: ' + D.bench[r.m] : '',
        isOpen: open, expanded: String(open), panelId: 'panel-' + r.id, btn: open ? 'Hide' : 'Details',
        btnStyle: 'flex: 0 0 auto; font-size: 14px; min-height: 36px; padding: 6px 14px; border-radius: 6px; cursor: pointer; border: 1px solid #c9d3da; background: ' + (open ? '#eef2f5' : '#ffffff') + '; color: #19354a; font-weight: 500',
        toggle: function () { var o = Object.assign({}, s.open); o[r.id] = !open; self.setState({ open: o }); },
        why: why, hasChoices: !!r.choices, radioName: 'm-' + r.id,
        choices: (r.choices || []).map(function (c) { var k = r.m + '|' + c[0], t = testedKey(k); return { label: c[1], desc: c[3], checked: c[0] === r.chosen, tag: !D.cmd[k] ? 'No command' : (t ? 'Runs on example data' : 'Not yet run'), tagStyle: t ? chip('#e3f1e8', '#1f6b3d', ' margin-left: 4px;') : chip('#fbf0d9', '#7a5300', ' margin-left: 4px;'), pick: function () { var mm = Object.assign({}, s.methods); mm[r.id] = c[0]; self.setRoute({ methods: mm }, c[1] + ' selected.'); } }; }),
        inputs: ins, outputs: outs.map(function (o) { return { v: o }; }), doc: BASE + (DOC[r.m] || '')
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
    var blocks = route.map(function (r, i) {
      var head = '# Step ' + (i + 1) + '. ' + r.title + ' (' + (LABEL[r.m] || r.m) + ')';
      if (r.soon) return head + '\necho "Step ' + (i + 1) + ' has no command yet: see its notebook." >&2; exit 1';
      return head + '\n' + r.wfs.map(function (w) { var c = D.cmd[r.m + '|' + w]; if (!c) return '# loading…'; c = fmt(c, cluster); if (own) c = c.replace(/tests\/fixtures\/[^\s`'"\\]+/g, function (p) { return s.paths[p] || '<your ' + nice(p).replace(/ /g, '_') + '>'; }); return c; }).join('\n\n');
    });
    var script = route.length ? '#!/bin/bash\n# xQTL analysis route from the Workflow Builder\n# Protocol revision: ' + REV + '\n# ' + (own ? 'Your data' : 'Repository example data') + ' · ' + { local: 'runs here', slurm: 'SoS submits each step to SLURM', sge: 'SoS submits each step to SGE' }[s.runOn] + (soonSteps.length ? '\n# NOT READY: step ' + soonSteps.join(', ') + ' has no command yet; the script stops there.' : '') + '\nset -euo pipefail\n\n' + blocks.join('\n\n') : '# Choose an example or describe your study to build the route.';
    var setupOnce = [{ v: 'Clone the repository.' }, { v: 'Install the software environment with pixi (see Environment setup).' }];
    if (s.runOn !== 'local') setupOnce.push({ v: 'Copy code/cluster/' + s.runOn + '.yml to my_cluster.yml and set your account and queue.' });
    var setupRoute = [{ v: 'Open a terminal in the repository folder' + (s.runOn === 'local' ? '.' : ', on the cluster’s login node.') }];
    if (own) setupRoute.push({ v: missing ? 'Fill in the ' + missing + ' input path' + (missing > 1 ? 's' : '') + ' above; the script shows <your …> until you do.' : 'Check that each path above points at a readable file.' });
    else setupRoute.push({ v: 'The example files are already in the repository; nothing to download.' });
    setupRoute.push({ v: s.runOn === 'local' ? 'Run the script: every step runs on this computer, one after another.' : 'Run the script: SoS sends each step to ' + (s.runOn === 'slurm' ? 'SLURM' : 'SGE') + ' and waits for it. This page does not submit anything itself.' });
    var mainResults = [], inter = [];
    route.forEach(function (r, i) { r.wfs.forEach(function (w) { ((D.io[r.m + '|' + w] || {}).o || []).forEach(function (o) {
      if (MAIN[r.m] && !mainResults.some(function (x) { return x.m === r.m; })) mainResults.push(Object.assign({ m: r.m, n: i + 1, module: LABEL[r.m], title: MAIN[r.m][0], meaning: MAIN[r.m][1], next: MAIN[r.m][2], path: o }, outOf(r.m)));
      else if (!inter.some(function (e) { return e.path === o; }) && !mainResults.some(function (x) { return x.path === o; })) inter.push({ n: i + 1, path: o });
    }); }); });
    var exObj = EXAMPLES.filter(function (e) { return e.id === s.example; })[0] || EXAMPLES[0];
    var copyText = function (text, which) { copyToClipboard(text, function (ok) { self.setState({ copied: ok ? which : 'fail' }); setTimeout(function () { self.setState({ copied: '' }); }, 2000); }); };

    // 5. Catalog
    var qq = (s.q || '').toLowerCase();
    var catalog = CATALOG.map(function (g) {
      var items = g[2].map(function (m) {
        var ks = Object.keys(D.cmd).filter(function (k) { return k.indexOf(m[0] + '|') === 0; }), nt = ks.filter(testedKey).length, soon = !!SOON[m[0]];
        return { id: m[0], label: m[1], desc: m[2], doc: BASE + (DOC[m[0]] || ''), onRoute: !!onRoute[m[0]], soon: soon,
          status: soon ? 'Coming soon' : (ks.length ? nt + ' of ' + ks.length + ' run on example data' : 'See notebook'),
          statusStyle: soon || !ks.length ? chip('#eef2f5', '#63717d') : (nt === ks.length ? chip('#e3f1e8', '#1f6b3d') : chip('#fbf0d9', '#7a5300')) };
      }).filter(function (it) { if (qq && (it.label + ' ' + it.desc + ' ' + it.id).toLowerCase().indexOf(qq) < 0) return false; if (s.filter === 'route') return it.onRoute; if (s.filter === 'soon') return it.soon; return true; });
      return { label: g[0].toUpperCase(), items: items };
    }).filter(function (g) { return g.items.length; });
    var filters = [['all', 'All'], ['route', 'On my route'], ['soon', 'Coming soon']].map(function (f) { var on = s.filter === f[0]; return { label: f[1], pressed: String(on), pick: function () { self.setState({ filter: f[0] }); }, style: 'font-size: 13.5px; min-height: 34px; padding: 5px 12px; border-radius: 6px; cursor: pointer; border: 1px solid ' + (on ? '#19354a' : '#c9d3da') + '; background: ' + (on ? '#19354a' : '#ffffff') + '; color: ' + (on ? '#ffffff' : '#202b35') }; });

    var testedCount = Object.keys(D.cmd).filter(testedKey).length;
    var journey = [['Understand the workflow', 'Stages and what each module does', '#map-h'], ['Choose how to begin', 'Example data or your own', '#begin-h'], ['Review your route', route.length ? route.length + ' steps, in order' : 'Modules in order', '#r-h'], ['Generate commands', 'Local, SLURM or SGE', '#c-h']].map(function (j, i) {
      return { n: i + 1, label: j[0], sub: j[1], href: j[2], style: 'padding: 12px 14px; background: ' + (i === 0 ? '#f6f8fa' : '#ffffff') + '; border-right: 1px solid #dce3e8' };
    });

    return {
      testedCount: testedCount, journey: journey, moduleCount: Object.keys(LABEL).length, clusterDoc: BASE + 'running-on-a-cluster',
      stageNav: CATALOG.map(function (g) { return { label: g[0] }; }),
      mapViews: mapViews, mapNote: mapNote, map: map, hasSel: !!selVals.label, sel: selVals, closeSel: function () { self.setState({ selMod: '' }); },
      modes: modes, isExample: isExample, isOwn: isOwn, examples: examples,
      qPheno: qPheno, qStart: qStart, qGoals: qGoals, hasContextual: contextual.length > 0, contextual: contextual,
      adv: s.adv, advExpanded: String(s.adv), toggleAdv: function () { self.setState({ adv: !s.adv }); }, advLabel: s.adv ? 'Hide advanced study settings' : 'Advanced study settings', advCount: nonDefault ? nonDefault + ' changed' : 'genotype QC, cohorts, contexts', allOpts: allOpts,
      msg: s.msg, msgStyle: s.msg ? 'margin: 12px 0 0; padding: 8px 12px; border-radius: 6px; background: #fbf0d9; color: #7a5300; font-size: 14px' : 'margin: 0; height: 0; overflow: hidden',
      route: routeVals, routeEmpty: route.length === 0, routeSummary: route.length ? route.length + ' steps · ' + nTested + ' run on example data' : '',
      srcTabs: [['example', 'Example data'], ['own', 'My own data']].map(function (t) { var on = s.src === t[0]; return { label: t[1], pressed: String(on), style: segStyle(on), pick: function () { self.setState({ src: t[0], mode: t[0] }); } }; }),
      runOn: [['local', 'Local terminal'], ['slurm', 'SLURM'], ['sge', 'SGE']].map(function (p) { var on = s.runOn === p[0]; return { label: p[1], checked: on, style: pillStyle(on), pick: function () { self.setState({ runOn: p[0] }); } }; }),
      showCluster: s.runOn !== 'local', runOnName: s.runOn === 'slurm' ? 'SLURM' : 'SGE', runOnFile: s.runOn, nativeCmd: s.runOn === 'slurm' ? 'sbatch' : 'qsub',
      isOwnSrc: own, isExampleSrc: !own && route.length > 0, pathGroups: pathGroups, noPaths: own && pathFields.length === 0,
      pathStatus: missing ? missing + ' of ' + pathFields.length + ' still needed' : 'Complete', pathStatusStyle: missing ? chip('#fbf0d9', '#7a5300') : chip('#e3f1e8', '#1f6b3d'),
      setupOnce: setupOnce, setupRoute: setupRoute, incomplete: soonSteps.length > 0, incompleteText: 'Step ' + soonSteps.join(', ') + ' has no command in the builder yet, so the script stops before it. Choose another method for that step or run it from its notebook.',
      mainResults: mainResults, noResults: route.length === 0, inter: inter, hasInter: inter.length > 0, interCount: String(inter.length),
      hasProduces: mainResults.length > 0, produces: 'This analysis produces ' + (function (t) { t = t.map(function (x) { return x.charAt(0).toLowerCase() + x.slice(1); }); return t.length > 1 ? t.slice(0, -1).join(', ') + ' and ' + t[t.length - 1] : t[0]; })(mainResults.map(function (m) { return m.title; })) + '.',
      jumpOwn: function () { self.setState({ mode: 'own', src: 'own', msg: '' }); },
      tut: { title: exObj.title, items: [{ k: 'WHAT YOU LEARN', v: exObj.learn }, { k: 'PROVIDED INPUTS', v: exObj.given }, { k: 'EXPECTED RESULT', v: exObj.result }, { k: 'AFTERWARDS', v: exObj.after }] },
      script: script, fileName: own ? 'xqtl_route.sh' : 'xqtl_route_example.sh', srcLabel: own ? 'your data' : 'example data',
      copyScriptLabel: s.copied === 'script' ? 'Copied' : (s.copied === 'fail' ? 'Copy failed' : 'Copy all'), copyScript: function () { copyText(script, 'script'); },
      downloadScript: function () { try { var b = new Blob([script + '\n'], { type: 'text/x-shellscript' }); var a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = own ? 'xqtl_route.sh' : 'xqtl_route_example.sh'; a.click(); setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000); } catch (e) {} },
      ownNeed: String(userOrder.length), useWithMine: function () { self.setState({ mode: 'own', src: 'own', msg: 'Kept your route and methods. Fill in the input paths below.' }); },
      q: s.q, onSearch: function (e) { self.setState({ q: e.target.value }); }, filters: filters, catalog: catalog, catalogEmpty: catalog.length === 0
    };
  }
}

/* ---- Mount ---- */
// the template is XML so that <sc-for> stays inside table rows (an HTML parser would move it out)
var root = document.getElementById('xqb');
var tpl = new DOMParser().parseFromString(document.getElementById('xqb-tpl').textContent, 'application/xml').documentElement;
var app = new Component(), pending = false;
app.setState = function (patch) { Object.assign(app.state, patch); if (!pending) { pending = true; (window.requestAnimationFrame || setTimeout)(function () { pending = false; paint(); }); } };
function paint() {
  // keep focus, caret and open <details> across re-renders
  var a = document.activeElement, fid = a && root.contains(a) && a.id, sel = null;
  if (fid && typeof a.selectionStart === 'number') sel = [a.selectionStart, a.selectionEnd];
  var open = {}; Array.prototype.forEach.call(root.querySelectorAll('details[id]'), function (d) { open[d.id] = d.open; });
  var frag = document.createDocumentFragment();
  kids(tpl, [app.renderVals()], frag);
  root.replaceChildren(frag);
  Object.keys(open).forEach(function (id) { var d = document.getElementById(id); if (d) d.open = open[id]; });
  if (fid) { var f = document.getElementById(fid); if (f) { f.focus(); if (sel && f.setSelectionRange) try { f.setSelectionRange(sel[0], sel[1]); } catch (e) {} } }
}
// in-page links scroll here instead of sending the parent page to the builder's own URL
root.addEventListener('click', function (e) {
  var a = e.target.closest && e.target.closest('a[href^="#"]'); if (!a) return;
  var t = document.getElementById(a.getAttribute('href').slice(1)); e.preventDefault();
  if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
});
paint();
})();
