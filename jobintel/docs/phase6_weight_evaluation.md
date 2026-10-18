# Phase 6 - weight combinations of the hybrid recommender (judged-pool re-ranking)

**Relevance labels: `assistant_judged` - these are AI judgments, NOT human labels.
**Profiles are derived from the query text (no real users); candidates are the judged jobs of each query; collaborative filtering is off (no history).** DL signal ON (denoising autoencoder v2).

20 queries; signals active for all queries: ir, lm, dfr, passage, pagerank, dl, content, knowledge, skill_match, preference.  Strict P@10 / MRR use grade 2 as relevant; P@10 (lenient) counts grade >= 1 and is near its ceiling (random order of the pool already scores 0.93).

| configuration | queries with evidence | nDCG@10 | P@10 (grade>=1) | P@10 (grade 2) | MRR (grade 2) | nDCG@10 minus default [95% CI] | Holm-adj. p |
|---|---|---|---|---|---|---|---|
| default without content | 20/20 | 0.869 | 0.990 | 0.720 | 0.939 | +0.004 [-0.015, +0.022] | 1.000 |
| default without skill_match | 20/20 | 0.868 | 0.985 | 0.720 | 0.965 | +0.003 [-0.003, +0.013] | 1.000 |
| default hybrid | 20/20 | 0.864 | 0.980 | 0.720 | 0.939 | - | - |
| default without passage | 20/20 | 0.863 | 0.980 | 0.715 | 0.939 | -0.002 [-0.007, +0.002] | 1.000 |
| default without pagerank | 20/20 | 0.861 | 0.975 | 0.715 | 0.939 | -0.003 [-0.014, +0.008] | 1.000 |
| equal weights (all active) | 20/20 | 0.861 | 0.990 | 0.715 | 0.886 | -0.003 [-0.021, +0.014] | 1.000 |
| default without dfr | 20/20 | 0.860 | 0.975 | 0.710 | 0.939 | -0.004 [-0.012, +0.002] | 1.000 |
| default without ir | 20/20 | 0.860 | 0.975 | 0.715 | 0.939 | -0.005 [-0.014, +0.004] | 1.000 |
| default without lm | 20/20 | 0.857 | 0.975 | 0.705 | 0.939 | -0.007 [-0.016, -0.000] | 1.000 |
| default without knowledge | 20/20 | 0.857 | 0.975 | 0.705 | 0.939 | -0.008 [-0.020, +0.005] | 1.000 |
| only ir | 20/20 | 0.852 | 0.985 | 0.700 | 0.912 | -0.012 [-0.043, +0.020] | 1.000 |
| default without dl | 20/20 | 0.846 | 0.985 | 0.705 | 0.899 | -0.018 [-0.035, -0.002] | 0.820 |
| default without preference | 20/20 | 0.845 | 0.960 | 0.720 | 0.939 | -0.020 [-0.058, +0.000] | 1.000 |
| recommender signals only (content, knowledge, skill_match, preference, dl) | 20/20 | 0.839 | 0.960 | 0.725 | 0.835 | -0.026 [-0.067, +0.005] | 1.000 |
| only content | 18/20 | 0.825 | 0.955 | 0.710 | 0.864 | -0.039 [-0.089, -0.002] | 1.000 |
| only dl | 17/20 | 0.824 | 0.960 | 0.695 | 0.870 | -0.041 [-0.078, -0.007] | 0.523 |
| IR family only (ir, lm, dfr, passage, pagerank) | 20/20 | 0.818 | 0.955 | 0.720 | 0.868 | -0.046 [-0.133, +0.009] | 1.000 |
| only skill_match | 18/20 | 0.812 | 0.950 | 0.695 | 0.800 | -0.052 [-0.104, -0.013] | 0.193 |
| only dfr | 20/20 | 0.794 | 0.950 | 0.685 | 0.835 | -0.071 [-0.142, -0.019] | 0.193 |
| only passage | 20/20 | 0.785 | 0.945 | 0.690 | 0.826 | -0.079 [-0.168, -0.018] | 0.316 |
| only pagerank | 20/20 | 0.762 | 0.955 | 0.630 | 0.771 | -0.103 [-0.169, -0.042] | 0.062 |
| only lm | 20/20 | 0.757 | 0.950 | 0.640 | 0.797 | -0.108 [-0.207, -0.031] | 0.162 |
| only preference | 2/20 | 0.729 | 0.940 | 0.585 | 0.640 | -0.135 [-0.198, -0.074] | 0.012 |
| random order of the pool | 20/20 | 0.716 | 0.927 | 0.574 | 0.758 | -0.148 [-0.204, -0.095] | 0.002 |
| only knowledge | 20/20 | 0.628 | 0.890 | 0.505 | 0.575 | -0.236 [-0.323, -0.161] | 0.001 |

The p-values compare each configuration with the default over the same queries (paired sign-flip randomisation, Holm-adjusted over all comparisons).  A difference that is not supported means 'not enough evidence with 20 queries', not 'no effect'.  A configuration whose signals do not exist for a query (e.g. 'only content' for a query with no recognised skill) has no evidence there and ranks that query in a fixed random order; 'queries with evidence' says for how many queries it had any signal.  MRR is averaged over the queries that have at least one grade-2 job.  The default weights are provisional starting values, not claimed optimal.
