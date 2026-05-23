"""
Module 3: Isolated Execution Sandbox.

Provides secure, ephemeral Docker-based code execution for generated code.
Every execution runs in a fresh, network-isolated container that is
immediately destroyed after capture, minimizing the blast radius of
malicious or buggy generated code.
"""
