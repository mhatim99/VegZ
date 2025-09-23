# Contributing to VegZ

Thank you for your interest in contributing to VegZ! This document provides guidelines for contributing to the project.

## Ways to Contribute

- **Bug Reports**: Report bugs through GitHub issues
- **Feature Requests**: Suggest new functionality
- **Code Contributions**: Submit pull requests
- **Documentation**: Improve documentation and examples
- **Testing**: Add tests for existing functionality
- **Academic Collaboration**: Contribute ecological methods and algorithms

## Reporting Bugs

When reporting bugs, please include:

1. **Python version** and operating system
2. **VegZ version**
3. **Minimal reproducible example**
4. **Expected vs actual behavior**
5. **Error messages and stack traces**

### Bug Report Template

```markdown
**Environment:**
- Python version: 
- VegZ version:
- Operating system:

**Description:**
Brief description of the bug.

**Reproducible Example:**
```python
# Minimal code to reproduce the issue
import VegZ
# ... your code here
```

**Expected Behavior:**
What you expected to happen.

**Actual Behavior:**
What actually happened.

**Error Messages:**
```
# Include full error traceback
```
```

## Feature Requests

For new features, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the scientific motivation**
3. **Provide references** to relevant ecological methods
4. **Suggest implementation approach**
5. **Consider backwards compatibility**

### Feature Request Template

```markdown
**Feature Description:**
Brief description of the proposed feature.

**Scientific Motivation:**
Why is this feature needed from an ecological perspective?

**Literature References:**
- Author, A. (Year). Title. Journal.
- Author, B. (Year). Title. Journal.

**Proposed Implementation:**
How might this feature be implemented?

**Example Usage:**
```python
# Example of how the feature would be used
```

**Backwards Compatibility:**
How does this affect existing code?
```

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/mhatim99/VegZ.git
cd vegz
```

### 2. Create Development Environment

```bash
python -m venv vegz-dev
source vegz-dev/bin/activate  # On Windows: vegz-dev\Scripts\activate
pip install -e ".[dev,spatial,remote-sensing,fuzzy,interactive]"
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
pytest tests/test_core.py                    # Core functionality
pytest tests/test_clustering.py             # Clustering and elbow analysis
pytest tests/test_diversity.py              # Diversity analysis
pytest -m "not slow"                        # Skip slow tests
pytest -m "plotting"                        # Only plotting tests
```

### Run Tests with Coverage
```bash
pytest --cov=vegz --cov-report=html
```

## Code Style

We use Black for code formatting and follow PEP 8 guidelines.

### Formatting
```bash
black vegz/
```

### Linting
```bash
flake8 vegz/
```

### Type Checking
```bash
mypy vegz/
```

## Documentation

### Building Documentation
```bash
cd docs/
sphinx-build -b html . _build/html
```

### Documentation Guidelines

1. **Docstrings**: Use NumPy/SciPy docstring format
2. **Examples**: Include usage examples in docstrings
3. **Mathematical notation**: Use proper LaTeX formatting
4. **References**: Cite relevant scientific literature

#### Docstring Template

```python
def my_function(data: pd.DataFrame, parameter: str = 'default') -> Dict[str, Any]:
    """
    Brief description of the function.
    
    Longer description with more details about what the function does,
    including any mathematical formulations or ecological interpretations.
    
    Parameters
    ----------
    data : pd.DataFrame
        Description of the data parameter. Should be a site-by-species matrix
        with sites as rows and species as columns.
    parameter : str, optional
        Description of the parameter, by default 'default'
    
    Returns
    -------
    Dict[str, Any]
        Description of what is returned, including structure of nested
        dictionaries if applicable.
    
    Raises
    ------
    ValueError
        If input data is invalid
    
    Notes
    -----
    Any additional notes about the method, including mathematical details
    or ecological interpretations.
    
    References
    ----------
    .. [1] Author, A. (Year). Title. Journal Name, Volume(Issue), pages.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from VegZ import MyClass
    >>> data = pd.DataFrame(...)
    >>> result = my_function(data, parameter='custom')
    >>> print(result.keys())
    ['key1', 'key2', 'key3']
    """
```

## Contributing Ecological Methods

When contributing new ecological methods:

### 1. Scientific Validation
- **Literature support**: Method should be published in peer-reviewed journals
- **Mathematical correctness**: Verify formulations and implementations
- **Ecological relevance**: Explain ecological interpretation

### 2. Implementation Guidelines
- **Test with known datasets**: Use datasets with known results
- **Handle edge cases**: Empty sites, single species, etc.
- **Performance considerations**: Optimize for large datasets
- **Parameter validation**: Check input parameters

### 3. Method Documentation
- **Algorithm description**: Explain the method clearly
- **Assumptions and limitations**: Be transparent about constraints
- **Interpretation guidelines**: Help users understand results
- **Literature references**: Cite original papers

## Pull Request Process

### 1. Before Submitting
- [ ] Tests pass locally
- [ ] Code is formatted with Black
- [ ] Documentation is updated
- [ ] Changelog is updated
- [ ] No merge conflicts

### 2. Pull Request Guidelines
- **Descriptive title**: Clearly describe the changes
- **Detailed description**: Explain what and why
- **Link related issues**: Reference issue numbers
- **Small, focused changes**: Easier to review

### 3. Pull Request Template
```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issues
Fixes #(issue number)

## Testing
- [ ] Tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed

## Documentation
- [ ] Documentation updated
- [ ] Examples added/updated
- [ ] Changelog updated

## Screenshots (if applicable)
```

## Ecological Code Review Criteria

When reviewing ecological code contributions:

### Scientific Accuracy
- [ ] Method implemented correctly
- [ ] Mathematical formulations accurate
- [ ] Appropriate for ecological data
- [ ] Edge cases handled properly

### Code Quality
- [ ] Well-structured and readable
- [ ] Appropriate comments and documentation
- [ ] Error handling and validation
- [ ] Performance considerations

### Testing
- [ ] Comprehensive test coverage
- [ ] Tests include ecological edge cases
- [ ] Performance tests for large datasets
- [ ] Comparison with known results

### Documentation
- [ ] Clear method description
- [ ] Usage examples provided
- [ ] Ecological interpretation explained
- [ ] Literature references included

## Academic Collaboration

We welcome collaborations with researchers and academics:

### Research Projects
- Implement new ecological methods
- Validate existing implementations
- Contribute benchmark datasets
- Write tutorials and case studies

### Publication Opportunities
- Methods papers describing new implementations
- Application papers using VegZ
- Comparison studies with other tools
- Educational materials and tutorials

### Contact for Collaboration
- Email: [research collaboration email]
- Create a GitHub discussion for public collaboration ideas

## Getting Help

- **GitHub Discussions**: For general questions and ideas
- **GitHub Issues**: For bugs and feature requests
- **Email**: For private inquiries
- **Documentation**: Check docs first

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Academic publications where appropriate
- Conference presentations

## Code of Conduct

Please be respectful and constructive in all interactions. I aim to create a welcoming environment for all contributors, regardless of background or experience level.

---

Thank you for contributing to VegZ!