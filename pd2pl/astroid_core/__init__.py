"""
astroid_core module - Astroid-based implementation of pandas to polars translation.

This module provides an alternative implementation of the translation logic using the
astroid library instead of the standard ast module, offering better support for
complex method chains and type inference.
"""

from pd2pl.astroid_core.transformer import AstroidBasedTransformer

__all__ = ['AstroidBasedTransformer']
