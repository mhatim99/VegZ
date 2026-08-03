"""
Import VegZ and report any warning VegZ itself raised while doing so.

Run by both tests/test_scientific_correctness.py and the "Importing VegZ must
be silent" step of the CI workflow, so the rule deciding which warnings belong
to VegZ lives in one place rather than being restated in YAML.

Exits 0 when VegZ imported without a warning of its own, and 1 otherwise,
listing the offenders on stderr.
"""

import pathlib
import sys
import warnings


def vegz_import_warnings():
    """
    Return the UserWarnings raised from VegZ's own source while importing it.

    Restricted to UserWarning because that is the category a library uses to
    tell a user something, and the one this check has always been about.
    VegZ's own DeprecationWarnings and FutureWarnings are not ignored - the
    filterwarnings settings in pyproject.toml already turn those into errors
    for the whole test run, so widening this check to every category would
    only duplicate that while making it fail on deprecations that third-party
    code raises with a stacklevel pointing back into VegZ.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        import VegZ

    package_dir = pathlib.Path(VegZ.__file__).resolve().parent

    def raised_by_vegz(record):
        # Ownership is decided by the warning's source file sitting inside the
        # installed package directory. Promoting every UserWarning to an error
        # instead makes this a test of the whole dependency tree - on Python
        # 3.9 the newest matplotlib that still installs there calls pyparsing
        # functions pyparsing now warns about, which is nothing VegZ can act
        # on. Matching the string 'VegZ' against the path is not good enough
        # either: a checkout or virtualenv can itself sit under a directory of
        # that name, which makes every third-party warning look like ours.
        try:
            return package_dir in pathlib.Path(record.filename).resolve().parents
        except (OSError, ValueError):
            return False

    return [w for w in caught
            if issubclass(w.category, UserWarning) and raised_by_vegz(w)]


def main():
    offenders = vegz_import_warnings()

    for record in offenders:
        print(f'{record.filename}:{record.lineno}: '
              f'{record.category.__name__}: {record.message}', file=sys.stderr)

    if offenders:
        print(f'{len(offenders)} warning(s) raised by VegZ on import',
              file=sys.stderr)
        return 1

    import VegZ
    print(f'VegZ {VegZ.__version__} imported without a warning of its own')
    return 0


if __name__ == '__main__':
    sys.exit(main())
