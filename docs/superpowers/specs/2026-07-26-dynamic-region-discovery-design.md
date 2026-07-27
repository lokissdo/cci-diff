# Dynamic Region Discovery Design

## Purpose

Remove the fixed number and identity of intervention regions from
counterfactual influence discovery. Each target label receives a
label-specific candidate set selected from source Grad-CAM++ evidence, semantic
mask area, and dataset support. Diffusion interventions then determine which
proposed set is sufficient to flip the generation classifier.

## Screening Evidence

For discovery image \(i\), Grad-CAM++ saliency \(A_i\), and semantic region
mask \(M_{ir}\), define the saliency coverage of region set \(S\):

\[
C_i(S)=
\frac{\sum_p A_i(p)\mathbf{1}[p\in\cup_{r\in S}M_{ir}]}
     {\max(\sum_p A_i(p),\epsilon)}.
\]

Define its semantic area:

\[
B_i(S)=\frac{|\cup_{r\in S}M_{ir}|}{HW}.
\]

A region is eligible only when its mask is present for every discovery image.
This keeps paired intervention sample counts identical. Evaluate every
non-empty subset of eligible regions up to eight members. A subset passes
screening when \(C_i(S)\) reaches the configured saliency threshold on at
least the configured fraction of discovery images.

Among passing subsets, select lexicographically by:

1. lowest mean semantic area;
2. fewest regions;
3. highest coverage frequency;
4. highest mean saliency coverage;
5. canonical region tuple.

If no subset passes, select the subset with the highest coverage frequency,
then highest mean coverage, lowest area, fewest regions, and canonical tuple.
The fallback is recorded and must not be described as satisfying the
screening criterion.

The default thresholds are protocol-level settings shared across labels:
saliency coverage \(0.80\), discovery-image frequency \(0.80\), and at most
eight selected regions. They determine evidence quality, not a fixed graph
size. Threshold sensitivity must be reported when used in a paper.

## Progressive Intervention Verification

For the dynamically proposed \(K\) regions, run same-seed masked diffusion
sets in increasing cardinality:

1. all singleton sets;
2. all pairs if no singleton reaches the required flip rate;
3. progressively larger sets;
4. stop after the first cardinality containing a set whose complete-cohort
   generation-classifier flip rate reaches \(0.95\);
5. otherwise exhaust sets through \(K\).

Stopping occurs only after all discovery images and seeds for one cardinality
finish. Incomplete sets cannot trigger stopping. Resuming validates existing
audits and re-evaluates the same stopping condition.

## Held-Out Use

The frozen graph and verified singleton regions are passed to the existing
source-only individual selector. The held-out image contributes no generated
output to region selection. There is one generation per held-out image, with
no escalation, reranking, or post-generation attack.

## Provenance

The screening manifest records the thresholds, eligible regions, all subset
metrics, selected set, selection status, and classifier digest. The
intervention manifest records planned and executed sets, per-cardinality flip
rates, stopping reason, and completion state.

## Non-Claims

Grad-CAM++ supplies classifier localization evidence, not causal proof.
Verified edges describe counterfactual influence for the frozen classifier,
dataset, masks, and diffusion intervention. They are not biological causal
relations.
