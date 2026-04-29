# Final Report Narrative Draft

## Results

We adapted the paper’s value-vector steering framework to SmolVLA and evaluated whether concept-aligned interventions causally affect behavior and whether those effects transfer across changing visual inputs. Our strongest results involve the `risk` and `fast` steering clusters.

In the Phase 5 alpha sweep, concept-aligned clusters changed behavior more than matched random neuron sets. This effect was especially strong for `risk`. At `alpha = 10`, the matched-random `risk` control had mean end-effector displacement `0.007345`, whereas the concept-aligned `risk` cluster reduced that value to `0.003435`. The `fast` cluster also beat its matched random control, but by a much smaller margin. These results indicate that at least some selected neuron sets are not merely large perturbations, but carry concept-specific causal influence on the policy’s behavior.

We then asked whether these behavioral effects transfer across fixed visual contexts. In Phase 6, we held the task and language instruction fixed while varying the init-state context. The `risk` cluster remained stable across all four paired init states, while `fast` was stable on three and near-collapse on one. This shows that transfer is not all-or-nothing: some internal steering directions behave like robust control features, while others are real but more context-sensitive.

Next, we tested transfer under nuisance visual perturbations applied to the primary camera stream. Under a mild brightness shift, both `risk` and `fast` remained stable overall, and the `risk` effect stayed especially strong. Under occlusion, both concepts still transferred, but `fast` weakened while `risk` remained stable and slightly strengthened. Taken together, these results suggest that the `risk` cluster behaves like a robust internal control direction in SmolVLA, whereas `fast` is weaker and more fragile.

The most important empirical conclusion is that some mechanistic steering directions in SmolVLA are not purely scene-specific. In particular, the `risk` cluster consistently outperformed matched random controls and preserved its behavioral effect across paired init states, brightness perturbations, and occlusion.

## Interpretation

Our interpretation is that SmolVLA contains at least some portable internal control features that can be identified through value-vector analysis and manipulated through FFN activation interventions. The strongest evidence supports this claim for `risk`. By contrast, `fast` appears to be a genuine but weaker direction whose effect depends more on context. This difference between `risk` and `fast` is itself informative: transfer is cluster-dependent, not guaranteed.

An important caution is that our semantic labels are operational rather than absolute. They arise from token-space decoding and clustering, not from perfect ground-truth disentanglement. This matters especially for weaker concepts. In addition, in this task the `fast` cluster does not literally make the arm move faster in an intuitive sense; instead, it reduces mean displacement relative to its matched random control. For this reason, our strongest claims should be about consistent causal influence rather than naive label semantics.

## SmolVLA-Specific Divergences From the Paper

Our reproduction preserves the paper’s core FFN value-vector logic, including intervention at the correct pre-`down_proj` hook point, but it diverges from the original setting in one major way: SmolVLA is a continuous-action VLM-plus-expert policy rather than an action-token VLA. As a result, our strongest evidence is not direct steering of action-token probabilities, but causal changes in rollout behavior and the transfer of those effects across visual contexts.

We also rely on a more automated semantic-labeling pipeline than a fully manual annotation workflow. We decode value vectors into token space, build semantic embeddings, cluster them, and rank clusters against concept anchors. This is scalable and practical, but it means our concept labels should be treated as approximations rather than perfect semantic ground truth.

Finally, our clearest current results are in motion statistics rather than task success. Success rates remain low in the short-horizon settings used for these experiments, so this project should be interpreted first as a mechanistic-control result and only secondarily as a task-performance result.

## Limitations

The current evaluation is still narrower than a full-scale paper-style benchmark reproduction. Most experiments are concentrated on one LIBERO task, rollout counts remain modest, and we have not yet run an all-task sweep or an exhaustive cluster-size sweep. In addition, we have not yet identified the perturbation boundary at which transfer breaks. Brightness and occlusion preserved the main effects we measured, especially for `risk`, so stronger perturbation sweeps are still needed to map the failure regime.

## Conclusion

This project demonstrates that the paper’s mechanistic steering framework can be meaningfully adapted to SmolVLA. We extracted FFN value vectors, assigned them approximate semantics through token-space decoding and clustering, intervened at the correct FFN activation point, and showed that some resulting steering directions causally affect behavior more than matched random controls. Most importantly, the strongest of these directions, `risk`, transfers across changing visual contexts and nuisance perturbations. This provides evidence that at least some internal control directions in a vision-language-action model are portable rather than purely scene-specific.
