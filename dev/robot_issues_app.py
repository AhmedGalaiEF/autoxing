import json

import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import pandas as pd

# Columns for the “database”
COLUMNS = [
    "id",
    "robot_type",
    "robot_manufacturer",
    "expected_behaviour",
    "actual_behaviour",
    "error_messages",
    "solutions",
]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Robot Issue Database"

app.layout = dbc.Container(
    [
        html.H2("Robot Issue Knowledge Base", className="mt-3 mb-4"),

        # Download component for JSON export
        dcc.Download(id="download-json"),

        dbc.Row(
            [
                # ------------------ LEFT: FORM ------------------
                dbc.Col(
                    [
                        html.H4("Add / Update Entry"),
                        dbc.Form(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    dbc.Label("Robot Type"),
                                                    dbc.Input(
                                                        id="input-robot-type",
                                                        placeholder="e.g. AMR, forklift, disinfection bot",
                                                        type="text",
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            md=6,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    dbc.Label("Manufacturer"),
                                                    dbc.Input(
                                                        id="input-robot-manufacturer",
                                                        placeholder="e.g. AutoXing, Unitree, etc.",
                                                        type="text",
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            md=6,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Expected behaviour"),
                                        dbc.Textarea(
                                            id="input-expected",
                                            placeholder="What the robot is supposed to do in this scenario...",
                                            style={"height": "80px"},
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Actual behaviour"),
                                        dbc.Textarea(
                                            id="input-actual",
                                            placeholder="What the robot actually did...",
                                            style={"height": "80px"},
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Error messages (logs, UI, codes)"),
                                        dbc.Textarea(
                                            id="input-errors",
                                            placeholder="Any error codes, logs, UI messages, etc.",
                                            style={"height": "80px"},
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Solutions / Workarounds"),
                                        dbc.Textarea(
                                            id="input-solutions",
                                            placeholder="Root cause, fix, workaround, configuration change...",
                                            style={"height": "80px"},
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Button(
                                                "Add entry",
                                                id="btn-add",
                                                color="primary",
                                                className="mt-2",
                                                n_clicks=0,
                                                style={"width": "100%"},
                                            ),
                                            md=4,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                id="form-status-msg",
                                                className="mt-3 text-success",
                                            ),
                                            md=8,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    md=5,
                ),

                # ------------------ RIGHT: TABLE ------------------
                dbc.Col(
                    [
                        html.H4("Database"),
                        dash_table.DataTable(
                            id="db-table",
                            columns=[{"name": col, "id": col} for col in COLUMNS],
                            data=[],  # this *is* the database
                            filter_action="native",
                            sort_action="native",
                            sort_mode="multi",
                            page_action="native",
                            page_size=10,
                            editable=True,          # cells editable
                            row_deletable=True,     # little “x” for each row
                            row_selectable="single",
                            selected_rows=[],
                            style_table={
                                "overflowX": "auto",
                                "maxHeight": "80vh",
                            },
                            style_cell={
                                "whiteSpace": "pre-line",
                                "height": "auto",
                                "textAlign": "left",
                                "fontFamily": "monospace",
                                "fontSize": "12px",
                            },
                            style_header={
                                "fontWeight": "bold",
                                "backgroundColor": "#f7f7f7",
                            },
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Download JSON",
                                        id="btn-download",
                                        color="secondary",
                                        className="mt-3",
                                        n_clicks=0,
                                        style={"width": "100%"},
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Delete selected row",
                                        id="btn-delete",
                                        color="danger",
                                        className="mt-3",
                                        n_clicks=0,
                                        style={"width": "100%"},
                                    ),
                                    md=6,
                                ),
                            ],
                            className="mt-2",
                        ),
                    ],
                    md=7,
                ),
            ],
            className="mt-3",
        ),
    ],
    fluid=True,
)

# -----------------------------------------------------------------------------
# Single callback for ADD and DELETE (no duplicate outputs)
# -----------------------------------------------------------------------------
@app.callback(
    Output("db-table", "data"),
    Output("form-status-msg", "children"),
    Output("input-robot-type", "value"),
    Output("input-robot-manufacturer", "value"),
    Output("input-expected", "value"),
    Output("input-actual", "value"),
    Output("input-errors", "value"),
    Output("input-solutions", "value"),
    Input("btn-add", "n_clicks"),
    Input("btn-delete", "n_clicks"),
    State("db-table", "data"),
    State("db-table", "selected_rows"),
    State("input-robot-type", "value"),
    State("input-robot-manufacturer", "value"),
    State("input-expected", "value"),
    State("input-actual", "value"),
    State("input-errors", "value"),
    State("input-solutions", "value"),
    prevent_initial_call=True,
)
def add_or_delete(
    n_add,
    n_delete,
    table_data,
    selected_rows,
    robot_type,
    robot_manufacturer,
    expected,
    actual,
    errors,
    solutions,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    data = table_data or []

    # --------- DELETE path ----------
    if trigger == "btn-delete":
        if not data or not selected_rows:
            msg = "No row selected to delete."
            # return unchanged form fields
            return (
                data,
                msg,
                robot_type,
                robot_manufacturer,
                expected,
                actual,
                errors,
                solutions,
            )

        idx = selected_rows[0]
        if 0 <= idx < len(data):
            deleted_id = data[idx].get("id")
            data.pop(idx)
            msg = f"Deleted row with id={deleted_id}."
        else:
            msg = "Invalid selection."

        return (
            data,
            msg,
            robot_type,
            robot_manufacturer,
            expected,
            actual,
            errors,
            solutions,
        )

    # --------- ADD path ----------
    # Minimal guard: require at least one non-empty field
    has_content = any(
        [
            robot_type,
            robot_manufacturer,
            expected,
            actual,
            errors,
            solutions,
        ]
    )
    if not has_content:
        return (
            data,
            "Nothing to add.",
            robot_type,
            robot_manufacturer,
            expected,
            actual,
            errors,
            solutions,
        )

    # Compute next ID from existing table
    if data:
        max_id = max([int(row.get("id", 0) or 0) for row in data])
    else:
        max_id = 0
    next_id = max_id + 1

    new_row = {
        "id": next_id,
        "robot_type": robot_type or "",
        "robot_manufacturer": robot_manufacturer or "",
        "expected_behaviour": expected or "",
        "actual_behaviour": actual or "",
        "error_messages": errors or "",
        "solutions": solutions or "",
    }
    data.append(new_row)

    msg = f"Entry #{new_row['id']} added."

    # Clear form
    return (
        data,
        msg,
        "",
        "",
        "",
        "",
        "",
        "",
    )


# -----------------------------------------------------------------------------
# Download JSON
# -----------------------------------------------------------------------------
@app.callback(
    Output("download-json", "data"),
    Input("btn-download", "n_clicks"),
    State("db-table", "data"),
    prevent_initial_call=True,
)
def download_json(n_clicks, table_data):
    data = table_data or []
    content = json.dumps(data, indent=2, ensure_ascii=False)
    return {
        "content": content,
        "filename": "robot_issues.json",
        "type": "text/json",
    }


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
