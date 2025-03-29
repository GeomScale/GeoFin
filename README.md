# PorQua - Portfolio Optimization and Quantitative Analysis

## Test Framework

This project uses a comprehensive test framework built with pytest, ensuring high code quality and reliability.

### Test Structure

```
test/
├── conftest.py           # Shared fixtures and configuration
├── test_quadratic_program.py  # Core optimization tests
└── ...
```

### Running Tests

1. Install test dependencies:
   ```bash
   pip install -r requirements-test.txt
   ```

2. Run tests with coverage:
   ```bash
   pytest --cov=src --cov-report=term-missing
   ```

3. Run specific test categories:
   ```bash
   # Unit tests only
   pytest -m unit
   
   # Integration tests
   pytest -m integration
   
   # Performance tests
   pytest -m performance
   
   # Property-based tests
   pytest -m property
   ```

### Quality Gates

The following quality gates are enforced:

- Minimum 90% code coverage
- Type safety (mypy --strict)
- Code formatting (black, isort)
- Linting (flake8)
- Performance benchmarks

### Test Categories

1. **Unit Tests** (`@pytest.mark.unit`)
   - Test individual components in isolation
   - Fast execution
   - No external dependencies

2. **Integration Tests** (`@pytest.mark.integration`)
   - Test component interactions
   - May use external dependencies
   - Slower execution

3. **Performance Tests** (`@pytest.mark.performance`)
   - Benchmark critical operations
   - Track performance over time
   - Generate performance reports

4. **Property-Based Tests** (`@pytest.mark.property`)
   - Test invariants and properties
   - Use hypothesis for test data generation
   - Comprehensive edge case coverage

### CI/CD Pipeline

The GitHub Actions workflow (`/.github/workflows/test.yml`) runs:

1. Type checking
2. Linting
3. Test suite with coverage
4. Performance benchmarks
5. Codecov integration

### Contributing

1. Create a feature branch:
   ```bash
   git checkout -b feat/tests/[component]-[feature]
   ```

2. Write tests first (TDD approach)

3. Run pre-commit checks:
   ```bash
   pytest --cov=src --cov-fail-under=90
   mypy --strict src/
   ```

4. Create a PR with:
   - Issue reference (`Fixes #XYZ`)
   - Codecov differential report
   - Type annotation documentation

### Maintenance

- Monthly dependency updates
- Regular benchmark baseline updates
- Legacy test deprecation cycle
- Performance regression monitoring

## License

[Your License Information]
