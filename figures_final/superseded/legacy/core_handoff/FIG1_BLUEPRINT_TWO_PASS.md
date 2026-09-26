# Blueprint 2 — TWO-PASS-CENTRIC  (spatial logic only, not a design)

The two passes are the organising idea; each gets its own lane.

```
              PASS 1  DIAGNOSE                    |  PASS 2  INTERVENE
  ------------------------------------------------+---------------------------
   [req+tools]                                    |  [same req+tools]
        |                                         |        |
        v                                         |        v
   [B1 .. B21 .. B26* .. B35]                     |  [B1 .. B21* .. B26 .. B35]
                    |  tap h_obs                  |        ^  h+alpha*d_c
                    v                             |        |
              (cached to disk)                    |        |
                    |                             |        |
              [router r_c]  --- p_c >= tau ? -----+--------+  fire
                                                  |        |
                                                  |        v
                                                  |   [a-hat'] -> 3 destinations
                                                  |             -> preservation
  ------------------------------------------------+---------------------------
   NOTE: L_inj(21) executes BEFORE L_obs(26). Router runs OUTSIDE both passes.
```

Strength: the impossible-one-pass error becomes visually impossible to make.
Weakness: costs horizontal space; the claim layer must go below or on a second row.
