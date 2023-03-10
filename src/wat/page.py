#!/usr/bin/python
#
# @brief  View classes for displaying whole websites.
# @author Luis Carlos Garcia-Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   13 Nov 2020.

from dash import html
import dash_bootstrap_components as dbc

# My imports
import wat.views.navbar
import wat.views.dashboard


class DashboardPage(object):
    """@class that displays the whole page."""
    def __init__(self, args, show_instructions=False):
        """@param[in]  args  Command line argparse args."""
        self.args = args
        self.show_instructions = show_instructions
    
    def generate_html(self):
        nav_view = wat.views.navbar.NavbarView()
        dashboard_view = wat.views.dashboard.DashboardView(self.args, self.show_instructions)
        content = html.Div([
            nav_view.generate_html(),
            dashboard_view.generate_html(),
        ])
        return content


if __name__ == '__main__':
    raise RuntimeError('[ERROR] This module cannot be run like a script.')
