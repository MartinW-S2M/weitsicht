Metadata Results
================

.. currentmodule:: weitsicht.metadata.metadata_results

The metadata helper functions return small result objects:

- Success results are dataclasses with ``ok=True``.
- Failures are :class:`~weitsicht.utils.ResultFailure` with ``ok=False`` and fields ``error`` and ``issues``.

.. rubric:: Issue Codes
.. autoclass:: MetadataIssue
   :members:

.. rubric:: IOR Results
.. autosummary::
   IORFromMetaResultSuccess
   IORFromMetaResult

.. autoclass:: IORFromMetaResultSuccess
.. autodata:: IORFromMetaResult

.. rubric:: EOR Results
.. autosummary::
   EORFromMetaResultSuccess
   EORFromMetaResult

.. autoclass:: EORFromMetaResultSuccess
.. autodata:: EORFromMetaResult

.. rubric:: Image Results
.. autosummary::
   ImageFromMetaResultSuccess
   ImageFromMetaResult

.. autoclass:: ImageFromMetaResultSuccess
.. autodata:: ImageFromMetaResult
