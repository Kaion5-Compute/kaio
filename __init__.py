"""Kaio Platform Python Client

A Python client for the Kaio multi-tenant machine learning platform that enables
developers to run SageMaker jobs through simple APIs with automatic image resolution
and secure file uploads.
"""

from .client import Client

__version__ = "0.1.5"
__all__ = ["Client"]
