"""
Core primitives: schemas, types, exceptions, and shared async infrastructure.

All domain models inherit from Pydantic v2 BaseModel for runtime validation,
serialization safety, and JSON Schema generation. This enforces structural
correctness at system boundaries (agent I/O, persistence, wire protocols).
"""
