# Value Vector Pattern Summary

- Catalog: `/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/results/value_vector_catalog_top30.jsonl`
- Top-token limit: `30`
- Min support: `4`

## Overall

- Total vectors: `81920`
- Meaningful-pattern vectors: `10836`
- Semantic-guess vectors: `5525`
- Non-semantic-guess vectors: `5311`
- Action-token-present vectors: `0`
- Special-token-present vectors: `2029`

## Layer Summary

| layer | total | meaningful | semantic-ish | non-semantic-ish | action-like | special-token |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2560 | 90 | 50 | 40 | 0 | 72 |
| 1 | 2560 | 142 | 76 | 66 | 0 | 79 |
| 2 | 2560 | 116 | 62 | 54 | 0 | 66 |
| 3 | 2560 | 131 | 63 | 68 | 0 | 68 |
| 4 | 2560 | 145 | 80 | 65 | 0 | 57 |
| 5 | 2560 | 133 | 73 | 60 | 0 | 74 |
| 6 | 2560 | 134 | 70 | 64 | 0 | 72 |
| 7 | 2560 | 156 | 92 | 64 | 0 | 76 |
| 8 | 2560 | 153 | 91 | 62 | 0 | 55 |
| 9 | 2560 | 145 | 79 | 66 | 0 | 68 |
| 10 | 2560 | 165 | 97 | 68 | 0 | 81 |
| 11 | 2560 | 173 | 113 | 60 | 0 | 59 |
| 12 | 2560 | 203 | 113 | 90 | 0 | 71 |
| 13 | 2560 | 181 | 90 | 91 | 0 | 62 |
| 14 | 2560 | 160 | 87 | 73 | 0 | 70 |
| 15 | 2560 | 160 | 88 | 72 | 0 | 58 |
| 16 | 2560 | 202 | 105 | 97 | 0 | 60 |
| 17 | 2560 | 285 | 162 | 123 | 0 | 77 |
| 18 | 2560 | 216 | 109 | 107 | 0 | 80 |
| 19 | 2560 | 207 | 127 | 80 | 0 | 73 |
| 20 | 2560 | 304 | 168 | 136 | 0 | 83 |
| 21 | 2560 | 408 | 235 | 173 | 0 | 80 |
| 22 | 2560 | 459 | 264 | 195 | 0 | 62 |
| 23 | 2560 | 627 | 295 | 332 | 0 | 62 |
| 24 | 2560 | 553 | 276 | 277 | 0 | 62 |
| 25 | 2560 | 766 | 356 | 410 | 0 | 52 |
| 26 | 2560 | 789 | 378 | 411 | 0 | 46 |
| 27 | 2560 | 756 | 372 | 384 | 0 | 43 |
| 28 | 2560 | 757 | 395 | 362 | 0 | 37 |
| 29 | 2560 | 699 | 374 | 325 | 0 | 37 |
| 30 | 2560 | 739 | 352 | 387 | 0 | 38 |
| 31 | 2560 | 682 | 233 | 449 | 0 | 49 |

## Semantic Examples

| layer | vector | pattern | support | top tokens |
| --- | --- | --- | --- | --- |
| 0 | 0 | stem_family:patient | 5 | roleum, cipline, iosity, Patient, pickle,  abre,  Near,  patients, rance, atology |
| 0 | 10 | stem_family:test | 4 |  screen, 分,  Republic, posium,  variety,  tester,  test,  extract,  shared,  following |
| 0 | 258 | stem_family:word | 7 | avering, rowsiness, reatment, ward, lessly, cerity,  Longer, gering, ength,  Words |
| 0 | 284 | stem_family:range | 4 | allets, ges, ceptives, ten, ting,  once, range, olesc,  ranges, sel |
| 0 | 309 | stem_family:out | 4 | out, s, ing, sample,  out,  lamp, termediate, ologically,  eventually,  magist |
| 0 | 367 | stem_family:far | 4 |  far, uckingham, numerable, termedi,  him,  much, entimes,  fish,  Fish, ertical |
| 0 | 421 | stem_family:case | 5 | cases, age, case,  case, pat, ly,  Cases,  cases,  app,  fore |
| 0 | 457 | stem_family:target | 5 | od,  target, tlement,  of, available,  targets, ai, target,  available,  stores |
| 0 | 478 | stem_family:function | 5 | IMIT, ortium, ED, Function, wealth, onium,  Function, EDS, ier,  fame |
| 0 | 490 | stem_family:turn | 4 | berra, chism,  Craw,  Pull,  turn, oride,  turned, omavirus,  pull, runk |
| 0 | 546 | stem_family:other | 4 |  ,  others, kyo,  commission, others,  complement,    , Others, cdot, ary |
| 0 | 588 | stem_family:effective | 4 |  celeb, :`~,  Effectiveness, comings,  Identities,  satell, <filename>, stuffs, stones,  Thy |
| 0 | 606 | stem_family:ear | 4 | numerable, ployed, conciliation, opl, ?|,  same, ener,  Ear, errals,  plans |
| 0 | 610 | stem_family:forward | 4 | lysses, otte, assadors, olkien, uggage, forward, haft, ahead, usiness, emetery |
| 0 | 628 | stem_family:keep | 4 |  full,  so,  length,  way, 
,  charge, way, y, ry,  part |
| 0 | 648 | stem_family:line | 5 |  line,  Line, 1,  Ram, al,  sure, Line, line,  Western, 2 |
| 0 | 725 | stem_family:space | 5 |  space,  Space,  never, Space, space,  himself,  themselves, ular,  spaces,  itself |
| 0 | 727 | stem_family:how | 4 | inic,  c,  course,  how, eners, how,  otherwise, SCs,  How, stood |
| 0 | 844 | stem_family:cross | 4 |  cross, One,  one, one,  One,  ill, cerity,  world, pliance,  Cross |
| 0 | 875 | stem_family:well | 4 | well,  well,  off, ade, f, off,  Well,  sheet,  cous, Well |

## Non Semantic Examples

| layer | vector | pattern | support | top tokens |
| --- | --- | --- | --- | --- |
| 0 | 86 | prefix_family:prac | 4 | ing,  ↩,  μg, full,  appl, osen,  fined,  commerc, ocon, Practice |
| 0 | 122 | prefix_family:exch | 4 |  Mach,  interchange,  exchange, hips,  expect,  emerging,  rest, …], erating, erate |
| 0 | 177 | prefix_family:time | 5 |  times,  hours, theless, RY, ternoon, raines,  actually,  actual, EEE, actually |
| 0 | 182 | prefix_family:doub | 5 | pherd,  double,  pries,  entirety,  Double, ording,  owes,  personally, ️, double |
| 0 | 250 | prefix_family:mast | 4 | agger, ερ, master, ivid, dash, gment,  master,  mix, upiter,  Master |
| 0 | 394 | prefix_family:part | 4 |  particularly,  core,  specific,  as,  poor,  most,  giving, 
,  or,  side |
| 0 | 435 | prefix_family:size | 4 |  size,  hang,  slower,  slow,  heavy,  sized, arval,  new,  Hang,  Slow |
| 0 | 463 | prefix_family:admi | 4 | pliance, wrights,  administered,  bulge,  cav,  customary,  Ci,  rig, Os, ctive |
| 0 | 528 | prefix_family:eval | 9 |  Eval,  evaluation,  evaluations, eld, piece, expression,  Ub, iv,  expression, aligned |
| 0 | 670 | exact_repeat_family:g | 4 |  , a,  type,  end,  g, e,  power,  state,  S,  A |
| 0 | 686 | prefix_family:appr | 4 | otives, 个, athy, utrition, quist, ipple, uber, odder, theless, cept |
| 0 | 874 | exact_repeat_family:p | 4 |  two, g,  ",  term,  address,  both,  little,  p,  ,  E |
| 0 | 877 | exact_repeat_family:in | 4 |  in, t, in, ing, re,  be,  where, s, f,  double |
| 0 | 900 | prefix_family:adva | 4 | MetaInfo,  ingred,  Inj,  advanced,  aneur, +', aps,  advoc,  impregn,  adv |
| 0 | 1041 | prefix_family:resi | 4 |  resistance,  trip, �,  resist,  super, ively,  sac,  superpowers,  #,  stage |
| 0 | 1154 | prefix_family:inte | 5 | unicip, assa, intestinal, athetic, opolitan, astrous, iberal, outheastern,  cuc,  Intelligence |
| 0 | 1165 | prefix_family:mode | 7 | s,  model,  mere,  Model,  nond,  models,  modelled,  non,  modeling, es |
| 0 | 1304 | prefix_family:comp | 5 | __':,  Conservancy, __":, udd, ucc, clusively,  telesc,  surg, company,  OCLC |
| 0 | 1306 | prefix_family:appl | 4 |  my,  application,  both,  life,  use,  Am,  ,  (,  much,  * |
| 0 | 1415 | exact_repeat_family:j | 4 | er, on,  or,  p, m,  j,  k,  P,  S,  l |

## Action Like Examples

| layer | vector | pattern | support | top tokens |
| --- | --- | --- | --- | --- |
