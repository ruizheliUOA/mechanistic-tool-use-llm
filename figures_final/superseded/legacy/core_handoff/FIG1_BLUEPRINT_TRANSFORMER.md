# Blueprint 1 — TRANSFORMER-CENTRIC  (spatial logic only, not a design)

The backbone dominates. Claim layer is a thin band underneath.

```
+----------------------------------------------------------------------+
| OFFLINE  baseline errors -> topology -> channel c -> [router] [d_c]   |  small
+----------------------------------------------------------------------+
                                                                         
  PASS 1                                     tap
  [req] -> [B1|B2|..|B26*|..|B35] ------------+                          
                                              v                          
                                        [router r_c] -> [p_c >= tau]     
                                              |                          
  PASS 2          (same input)                | fire                     
  [req] -> [B1|..|B21*|..|B35] -> [a-hat']    |                          
                    ^-----------------------  +                          
                  h+alpha*d_c                                            
                                                                         
                          |                                              
                          +--> DESTINATIONS   SOURCE_RETAINED            
                          |                   GOLD_ARRIVAL               
                          |                   OTHER_WRONG                
                          +--> PRESERVATION   retained / broken          
                                                                         
+----------------------------------------------------------------------+
| Adjudicable -> [Readable -> Steerable -> Correctable] -> {Licensable} |
+----------------------------------------------------------------------+
```

Strength: mechanism is unmistakable; matches dense CV method-figure conventions.
Weakness: the claim layer reads as an afterthought, which understates the
contribution. Best if Figure 2 carries the claim structure instead.
