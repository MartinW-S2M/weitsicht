.. currentmodule:: weitsicht

----------------------------------------
Return Types for Mapping and Projection
----------------------------------------

:py:class:`Issue` holds the types which can be returned by issues as set

.. autoclass:: Issue
   :members:
   :exclude-members: __init__


:py:class:`ResultFailure` is the common Return Type for Failures,

.. autoclass:: ResultFailure
   :members:
   :exclude-members: __init__


Result for Mapping
----------------------------------------

.. autosummary:: MappingResult


:py:class:`MappingResultSuccess` Dataclass for successful mappings

.. autoclass:: MappingResultSuccess
   :members: False
   :exclude-members: __init__


Result for Projection
----------------------------------------

.. autosummary:: ProjectionResult


:py:class:`ProjectionResultSuccess` Dataclass for successful projections

.. autoclass:: ProjectionResultSuccess
   :members: False
   :exclude-members: __init__
