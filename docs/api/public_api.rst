Public API
==========

The following objects form the intentionally exported pre-release API.

The basic local interface is launched by ``selcal ui --workspace DIRECTORY``.
Its HTTP routes and saved UI observations are local application details, not
additional scientific Python API or calculation authority. The capability is
``PARTIAL_UI_BASIC_LOOP``. UI resume, record content checks, reports, evidence
export and downloads are implemented through the same application service.
Full integration and release validation remain separate gates. Saved content
checks are not historical execution authentication or scientific validation.

Contracts
---------

.. autoclass:: selcal.contracts.PlanRequest
   :members:

.. autoclass:: selcal.contracts_v2.PlanRequestV2
   :members:

.. autoclass:: selcal.contracts.ResolvedScientificPlan
   :members:

.. autoclass:: selcal.contracts_v2.ResolvedScientificPlanV2
   :members:

Resolution and migration
------------------------

.. autoclass:: selcal.resolution.PlanResolution
   :members:

.. autoclass:: selcal.resolution_v2.PlanResolutionV2
   :members:

.. autoclass:: selcal.migration_v1_to_v2.PlanMigrationV1ToV2
   :members:

.. autofunction:: selcal.resolution.resolve_plan

.. autofunction:: selcal.resolution_v2.resolve_plan_v2

.. autofunction:: selcal.migration_v1_to_v2.migrate_plan_v1_to_v2

Calibration and verification
----------------------------

``selcal.calibrate_selected_family`` is the entry for new analyses. Before any
computation it applies the same admission rule as the command line: a null that
SelCal does not support for inference raises ``InvalidNullForInferenceError``
(``code == "invalid_null_for_inference"``). The current policy refuses
``circular_shift_v2`` with ``min_shift`` greater than 1; it is a policy of
supported configurations, not a claim that every such shift set fails to be a
group. Other nulls are not judged by this rule.

.. autofunction:: selcal.inference.calibrate_selected_family

.. autoexception:: selcal.inference.InvalidNullForInferenceError

.. autofunction:: selcal.inference.null_supports_inference

.. autofunction:: selcal.calibration_v2.verify_calibration_result

.. code-block:: python

   import selcal
   from selcal.contracts import SeriesPair

   pair = SeriesPair(source=x, target=y)
   resolution = selcal.resolve_plan_v2(request)
   try:
       result = selcal.calibrate_selected_family(pair, resolution)
   except selcal.InvalidNullForInferenceError as error:
       print(error.code)  # "invalid_null_for_inference"

Internal kernel (not a supported analysis entry)
------------------------------------------------

``selcal.calibration_v2.calibrate_selected_family`` is the unrestricted
calibration kernel kept for record replay and internal tests. It does not apply
the admission rule and must not be used for new analyses.

Result record serialization (development API)
----------------------------------------------

These functions retain complete result content. Decoding does not authenticate
the input pair or historical execution and does not resume or replay a run.
The caller supplies an explicit per-record byte limit; it is not a process-memory
guarantee or a universal dataset limit.

.. autofunction:: selcal.result_wire.encode_calibration_result

.. autofunction:: selcal.result_wire.decode_calibration_result

.. autoclass:: selcal.result_wire.ResultWireError

File workflow (development API)
--------------------------------

These entrypoints share the same service as the installed ``selcal`` command.
Terminal reading and reporting do not imply checkpoint recovery or authentication.

.. autoclass:: selcal.workflow_config.WorkflowConfig
   :members:

.. autofunction:: selcal.workflow_config.decode_workflow_config

.. autofunction:: selcal.workflow_config.encode_workflow_config

.. autofunction:: selcal.workflow.validate_files

.. autofunction:: selcal.workflow.run_files

.. autofunction:: selcal.workflow.read_workflow

.. autofunction:: selcal.workflow.verify_record

.. autofunction:: selcal.workflow.report_record

.. autofunction:: selcal.workflow.doctor

Portable evidence (development API)
-----------------------------------

The exporter and verifier use retained content only. They neither replay
calibration nor authenticate historical execution. ``max_bytes`` bounds the
aggregate serialized bundle, including its manifest, not peak process memory.

.. autofunction:: selcal.workflow_export.export_record

.. autofunction:: selcal.workflow_export.verify_export

.. autoclass:: selcal.workflow_export.ExportError
