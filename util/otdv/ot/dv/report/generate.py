# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from jinja2 import Environment, PackageLoader, Template, select_autoescape

__all__ = ("generate_report",)


_REPORT_ENV: Optional[Environment] = None


def _get_report_template(name: str) -> Template:
    """Get a report template"""
    assert __package__ is not None, "must be defined within a python package"
    global _REPORT_ENV

    if _REPORT_ENV is None:
        _REPORT_ENV = Environment(
            loader=PackageLoader(__package__),
            autoescape=select_autoescape(enabled_extensions=("html",)),
        )

    return _REPORT_ENV.get_template(f"{name}.j2")


def generate_report() -> None:
    """Generate a report"""
    template = _get_report_template("sim_report.html")
    print(template.render())
