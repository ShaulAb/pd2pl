"""Tests for date transformation functionality."""

import pytest
import ast
from datetime import date, datetime
from pd2pl.datetime_utils.transformers import create_date_ast_node, transform_date_range, INCLUSIVE_TO_CLOSED

# Mock PandasToPolarsVisitor for testing
class MockVisitor:
    def visit(self, node):
        return node  # Just return the node unchanged for simple testing

def test_create_date_ast_node_date():
    """Test creating AST nodes for date objects."""
    test_date = date(2023, 1, 15)
    node = create_date_ast_node(test_date)
    
    # Check node structure
    assert isinstance(node, ast.Call)
    assert isinstance(node.func, ast.Name)
    assert node.func.id == 'date'
    assert len(node.args) == 3
    assert [arg.value for arg in node.args] == [2023, 1, 15]

def test_create_date_ast_node_datetime():
    """Test creating AST nodes for datetime objects."""
    test_datetime = datetime(2023, 1, 15, 12, 30, 45, 123456)
    node = create_date_ast_node(test_datetime)
    
    # Check node structure
    assert isinstance(node, ast.Call)
    assert isinstance(node.func, ast.Name)
    assert node.func.id == 'datetime'
    assert len(node.args) == 7
    assert [arg.value for arg in node.args] == [2023, 1, 15, 12, 30, 45, 123456]

def test_transform_date_range_basic():
    """Test basic transformation of date_range with start/end."""
    # Create a pandas date_range call AST
    pd_call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='pd', ctx=ast.Load()),
            attr='date_range',
            ctx=ast.Load()
        ),
        args=[],
        keywords=[
            ast.keyword(arg='start', value=ast.Call(
                func=ast.Name(id='date', ctx=ast.Load()),
                args=[ast.Constant(value=2023), ast.Constant(value=1), ast.Constant(value=1)],
                keywords=[]
            )),
            ast.keyword(arg='end', value=ast.Call(
                func=ast.Name(id='date', ctx=ast.Load()),
                args=[ast.Constant(value=2023), ast.Constant(value=1), ast.Constant(value=10)],
                keywords=[]
            )),
            ast.keyword(arg='freq', value=ast.Constant(value='D'))
        ]
    )
    
    # Transform to polars
    visitor = MockVisitor()
    pl_call = transform_date_range(pd_call, visitor)
    
    # Check the transformed call
    assert isinstance(pl_call, ast.Call)
    assert isinstance(pl_call.func, ast.Attribute)
    assert pl_call.func.attr == 'date_range'
    assert isinstance(pl_call.func.value, ast.Name)
    assert pl_call.func.value.id == 'pl'
    
    # Check keywords
    keywords = {kw.arg: kw.value for kw in pl_call.keywords}
    assert 'start' in keywords
    assert 'end' in keywords
    assert 'interval' in keywords
    assert isinstance(keywords['interval'], ast.Constant)
    assert keywords['interval'].value == '1d'

def test_transform_date_range_with_periods():
    """Test transformation of date_range with periods instead of end."""
    # Create a pandas date_range call AST with periods and freq
    pd_call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='pd', ctx=ast.Load()),
            attr='date_range',
            ctx=ast.Load()
        ),
        args=[],
        keywords=[
            ast.keyword(arg='start', value=ast.Call(
                func=ast.Name(id='date', ctx=ast.Load()),
                args=[ast.Constant(value=2023), ast.Constant(value=1), ast.Constant(value=1)],
                keywords=[]
            )),
            ast.keyword(arg='periods', value=ast.Constant(value=5)),
            ast.keyword(arg='freq', value=ast.Constant(value='D'))
        ]
    )
    
    # Transform to polars
    visitor = MockVisitor()
    pl_call = transform_date_range(pd_call, visitor)
    
    # Check the transformed call has end date calculated
    keywords = {kw.arg: kw.value for kw in pl_call.keywords}
    assert 'start' in keywords
    assert 'end' in keywords  # Should calculate end from periods
    assert 'interval' in keywords
    assert isinstance(keywords['interval'], ast.Constant)
    assert keywords['interval'].value == '1d'

def test_transform_date_range_with_inclusive():
    """Test transformation of date_range with inclusive parameter."""
    # Create pandas date_range calls with different inclusive values
    for inclusive, expected_closed in INCLUSIVE_TO_CLOSED.items():
        pd_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='pd', ctx=ast.Load()),
                attr='date_range',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[
                ast.keyword(arg='start', value=ast.Call(
                    func=ast.Name(id='date', ctx=ast.Load()),
                    args=[ast.Constant(value=2023), ast.Constant(value=1), ast.Constant(value=1)],
                    keywords=[]
                )),
                ast.keyword(arg='end', value=ast.Call(
                    func=ast.Name(id='date', ctx=ast.Load()),
                    args=[ast.Constant(value=2023), ast.Constant(value=1), ast.Constant(value=10)],
                    keywords=[]
                )),
                ast.keyword(arg='freq', value=ast.Constant(value='D')),
                ast.keyword(arg='inclusive', value=ast.Constant(value=inclusive))
            ]
        )
        
        # Transform to polars
        visitor = MockVisitor()
        pl_call = transform_date_range(pd_call, visitor)
        
        # Check the transformed call has correct closed parameter
        keywords = {kw.arg: kw.value for kw in pl_call.keywords}
        assert 'closed' in keywords
        assert isinstance(keywords['closed'], ast.Constant)
        assert keywords['closed'].value == expected_closed

def test_transform_date_range_with_dynamic_args():
    """Test transformation of date_range with non-constant arguments."""
    # Create a pandas date_range call AST with variable references
    pd_call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='pd', ctx=ast.Load()),
            attr='date_range',
            ctx=ast.Load()
        ),
        args=[],
        keywords=[
            ast.keyword(arg='start', value=ast.Name(id='start_date', ctx=ast.Load())),
            ast.keyword(arg='periods', value=ast.Name(id='n_periods', ctx=ast.Load())),
            ast.keyword(arg='freq', value=ast.Name(id='frequency', ctx=ast.Load()))
        ]
    )
    
    # Transform to polars
    visitor = MockVisitor()
    pl_call = transform_date_range(pd_call, visitor)
    
    # Check the transformed call has correct structure for dynamic args
    keywords = {kw.arg: kw.value for kw in pl_call.keywords}
    assert 'start' in keywords
    assert 'periods' in keywords  # Should pass through periods for dynamic args
    assert 'interval' in keywords
    assert isinstance(keywords['interval'], ast.Call)  # Should be a call to the mapping function 