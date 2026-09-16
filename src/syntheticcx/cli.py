"""Command-line identity for SyntheticCX.

Execution commands are intentionally deferred until configuration, runtime, and
persistence contracts exist.
"""

import typer

from syntheticcx import __version__

app = typer.Typer(
    add_completion=False,
    help="SyntheticCX: simulate customers and test conversational journeys.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Print the installed package version when requested."""
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()  # type: ignore[untyped-decorator]
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the installed SyntheticCX version and exit.",
    ),
) -> None:
    """Expose package help and version until runtime commands are implemented."""
