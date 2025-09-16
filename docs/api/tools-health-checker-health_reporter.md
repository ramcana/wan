---
title: tools.health-checker.health_reporter
category: api
tags: [api, tools]
---

# tools.health-checker.health_reporter

Health reporting and analytics system

## Classes

### HealthReporter

Generates comprehensive health reports with analytics and visualizations

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x0000019431A9D8A0>)



##### generate_report(self: Any, health_report: HealthReport, format_type: str) -> Path

Generate a formatted health report

Args:
    health_report: The health report data
    format_type: Output format ("html", "json", "markdown", "console")
    
Returns:
    Path to the generated report file

##### _generate_html_report(self: Any, report: HealthReport, timestamp: str) -> Path

Generate HTML report with dashboard-style layout

##### _generate_json_report(self: Any, report: HealthReport, timestamp: str) -> Path

Generate JSON report for programmatic access

##### _generate_markdown_report(self: Any, report: HealthReport, timestamp: str) -> Path

Generate Markdown report for documentation

##### _print_console_report(self: Any, report: HealthReport) -> None

Print a formatted console report

##### generate_dashboard_data(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019431A3BAF0>

Generate data for real-time dashboard

##### analyze_trends(self: Any, history_file: Path) -> <ast.Subscript object at 0x0000019434037370>

Analyze health trends from historical data

##### _calculate_volatility(self: Any, scores: <ast.Subscript object at 0x00000194340374F0>) -> float

Calculate score volatility (standard deviation)

##### _generate_trend_recommendations(self: Any, analysis: <ast.Subscript object at 0x0000019434037940>) -> <ast.Subscript object at 0x0000019431B62EF0>

Generate recommendations based on trend analysis

##### _get_html_styles(self: Any) -> str

Get CSS styles for HTML report

##### _get_score_class(self: Any, score: float) -> str

Get CSS class for score

##### _get_score_label(self: Any, score: float) -> str

Get human-readable score label

##### _get_score_emoji(self: Any, score: float) -> str

Get emoji for score

##### _get_console_color(self: Any, score: float) -> str

Get console color code for score

##### _generate_component_cards(self: Any, report: HealthReport) -> str

Generate HTML for component cards

##### _generate_issues_html(self: Any, report: HealthReport) -> str

Generate HTML for issues section

##### _generate_trends_html(self: Any, trends: HealthTrends) -> str

Generate HTML for trends section

##### _generate_component_details_html(self: Any, report: HealthReport) -> str

Generate HTML for component details

##### _generate_chart_scripts(self: Any, report: HealthReport) -> str

Generate JavaScript for charts

##### _generate_component_table_rows(self: Any, report: HealthReport) -> str

Generate table rows for component scores

##### _generate_issues_markdown(self: Any, issues: <ast.Subscript object at 0x000001942F29BEE0>) -> str

Generate markdown for issues list

##### _generate_trends_markdown(self: Any, trends: HealthTrends) -> str

Generate markdown for trends

##### _generate_component_details_markdown(self: Any, report: HealthReport) -> str

Generate markdown for component details

