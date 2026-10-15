# Source and comparison records

Records identify their tested source tree, file hashes and comparison scope.
An earlier record is evidence for that tree, not a claim about every later build.
Use [architecture](../architecture.md) for comparison methods, [performance](../performance.md)
for measurement conditions and [validation](../validation.md) for execution results.
Raw build archives and reference libraries are not distributed.

| Subject | Records |
|---|---|
| Sources and build recipes | [engines](engines.json), [additional_engines](additional_engines.json), [source_trees](source_trees.json) |
| K96 imports | [import_real](import_real.json), [import_real_crt](import_real_crt.json), [import_complex](import_complex.json), [import_cgemm](import_cgemm.json), [import_single](import_single.json), [import_small_single](import_small_single.json), [import_small_complex](import_small_complex.json), [import_integer](import_integer.json) |
| K64 imports and measurements | [import_k64](import_k64.json), [k64_calibration](k64_calibration.json), [k64_measurements](k64_measurements.json), [k64_cgemm_measured](k64_cgemm_measured.json) |
| Measurement and figure sources | [measurement_choices](measurement_choices.json), [performance_sources](performance_sources.json), [hardware_rates](hardware_rates.json), [figures](figures.json), [sparse_measurements](sparse_measurements.json) |
| Build and API checks | [cmake_validation](cmake_validation.json), [api_build_validation](api_build_validation.json), [api_contract_validation](api_contract_validation.json), [api_options_validation](api_options_validation.json), [module_isolation_validation](module_isolation_validation.json), [device_guard_validation](device_guard_validation.json) |
| Host settings and descriptions | [measured_defaults](measured_defaults.json), [measured_defaults_validation](measured_defaults_validation.json), [engine_info_validation](engine_info_validation.json), [engine_description_validation](engine_description_validation.json), [error_message_validation](error_message_validation.json), [info_relocation_validation](info_relocation_validation.json), [host_relocation_validation](host_relocation_validation.json) |
| Tables and reconstruction | [coverage_bounds](coverage_bounds.json), [strict_coverage_validation](strict_coverage_validation.json), [table_bounds](table_bounds.json), [table_bounds_validation](table_bounds_validation.json) |
| Source and comment checks | [source_cleanup_validation](source_cleanup_validation.json), [comment_review](comment_review.json), [comment_revision_validation](comment_revision_validation.json), [encoder_comment_validation](encoder_comment_validation.json), [fixup_validation](fixup_validation.json), [expanded_validation](expanded_validation.json), [patch_comment_validation](patch_comment_validation.json) |
| Shared kernels and geometry | [level3_support](level3_support.json), [support_copy_validation](support_copy_validation.json), [cgemm_geometry_validation](cgemm_geometry_validation.json) |
| Template comparisons | [traits_validation](traits_validation.json), [template_moduli_trial](template_moduli_trial.json), [d17_preprocessing_validation](d17_preprocessing_validation.json), [d18_template_validation](d18_template_validation.json), [template_import_validation](template_import_validation.json) |
| Documentation checks | [documentation_translation_validation](documentation_translation_validation.json), [performance_number_validation](performance_number_validation.json) |
| Directory layout | [layout_migration](layout_migration.json), [layout_validation](layout_validation.json) |
| Catalogue collection | [solution_collection](solution_collection.json) |
| Legacy data | [legacy_contracts](legacy_contracts.json), [legacy_moduli](legacy_moduli.json), [legacy_inclusion](legacy_inclusion.json) |

[GPU results from 14 October](../Results/GPU_RESULTS_20261014.json) and
[15 October](../Results/GPU_RESULTS_20261015.json) retain case-level evidence.

## Import records

`engines.json` includes source digests before and after patches, patch order,
build arguments and reference kinds. K96 measured-binary comparisons and K64
source comparisons have different limits; later Arm comparisons have their own
records. Missing timing-to-library links are stated in each import record.

[Patches](Patches/) preserve the import programs; they are not run during a build.
The SYRK patch precedes the TRMM patch. Applying either twice is unsupported.
[Original and edited patch hashes](Patches/comment_cleanup.json) and
[current comment checks](patch_comment_validation.json) identify their revisions.
Original patches remain in the research archive.

## Legacy data

[Legacy/20260922](../../Legacy/20260922/) retains the old tree outside the current
build. [Contracts](legacy_contracts.json) and [modulus sets](legacy_moduli.json)
contain numeric comparisons; [file roles and comparison rules](legacy_inclusion.json)
explain their scope. A corresponding role does not imply API compatibility.
