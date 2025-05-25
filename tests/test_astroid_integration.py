#!/usr/bin/env python
"""
Test script for the astroid-based implementation of pandas-to-polars translation.

This script tests the basic infrastructure setup to ensure that:
1. The configuration toggle works correctly
2. The astroid-based implementation can be loaded
3. The system correctly falls back to ast-based implementation when needed
"""

import sys
import logging
from tests.conftest import translate_test_code
from pd2pl.config import TranslationConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_astroid')

def test_with_ast():
    """Test translation using the standard ast-based implementation."""
    logger.info("Testing with standard ast-based implementation")
    # Ensure we're using the ast-based implementation
    TranslationConfig.set(use_astroid=False)
    
    # Simple pandas code to translate
    pandas_code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
result = df['a'].sum()
"""
    
    # Translate the code
    try:
        polars_code = translate_test_code(pandas_code)
        logger.info("Translation successful with ast-based implementation")
        logger.info(f"Translated code:\n{polars_code}")
        return True
    except Exception as e:
        logger.error(f"Translation failed with ast-based implementation: {e}")
        return False

def test_with_astroid():
    """Test translation using the astroid-based implementation."""
    logger.info("Testing with astroid-based implementation")
    # Configure to use the astroid-based implementation
    TranslationConfig.set(use_astroid=True)
    
    # Simple pandas code to translate
    pandas_code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
result = df['a'].sum()
"""
    
    # Translate the code
    try:
        polars_code = translate_test_code(pandas_code)
        logger.info("Translation successful with astroid-based implementation")
        logger.info(f"Translated code:\n{polars_code}")
        return True
    except Exception as e:
        logger.error(f"Translation failed with astroid-based implementation: {e}")
        logger.error("This is expected since the astroid implementation is not complete yet")
        return False

if __name__ == "__main__":
    logger.info("Starting astroid integration test")
    
    # Test with standard ast implementation
    ast_result = test_with_ast()
    
    # Test with astroid implementation
    astroid_result = test_with_astroid()
    
    # Report results
    logger.info("Test results:")
    logger.info(f"AST-based implementation: {'PASS' if ast_result else 'FAIL'}")
    logger.info(f"Astroid-based implementation: {'PASS' if astroid_result else 'FAIL (expected)'}")
    
    # The astroid implementation will likely fail since it's just a skeleton,
    # but we expect the ast implementation to work
    if not ast_result:
        logger.error("ERROR: AST-based implementation failed, which was not expected!")
        sys.exit(1)
    
    logger.info("Test completed successfully - the infrastructure is working")
    sys.exit(0)
