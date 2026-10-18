"""English stop-word list (embedded so nothing has to be downloaded).
Negations ('no', 'not', 'nor') are deliberately KEPT: 'no experience required' must not collapse to 'experience required'."""
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just let me more most my myself of off on once only or other
ought our ours ourselves out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself yourselves also etc via per within without across among upon
""".split()) - {"no", "not", "nor"}
