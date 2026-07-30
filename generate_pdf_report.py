#!/usr/bin/env python3
"""
PDF Report Generation for Premier League Prediction Model
Creates professional PDF reports with statistical significance analysis
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import pandas as pd
from datetime import datetime
import os
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

def create_feature_importance_chart(importance_df, top_n=10):
    """Create a horizontal bar chart of top feature importances"""
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(top_n)

    # Create color mapping based on significance
    colors_list = []
    for sig in top_features['Significance']:
        if '***' in sig:
            colors_list.append('#2E8B57')  # Dark green for highly significant
        elif '**' in sig:
            colors_list.append('#32CD32')  # Green for significant
        elif '*' in sig:
            colors_list.append('#FFD700')  # Gold for moderately significant
        elif '.' in sig:
            colors_list.append('#FFA500')  # Orange for weakly significant
        else:
            colors_list.append('#DC143C')  # Red for not significant

    bars = plt.barh(range(len(top_features)), top_features['Importance %'])
    for i, (bar, color) in enumerate(zip(bars, colors_list)):
        bar.set_color(color)

    plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=8)
    plt.xlabel('Importance (%)', fontsize=10)
    plt.title(f'Top {top_n} Feature Importances with Statistical Significance', fontsize=12, pad=20)
    plt.grid(axis='x', alpha=0.3)

    # Create legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#2E8B57', label='*** p < 0.001'),
        plt.Rectangle((0,0),1,1, facecolor='#32CD32', label='** p < 0.01'),
        plt.Rectangle((0,0),1,1, facecolor='#FFD700', label='* p < 0.05'),
        plt.Rectangle((0,0),1,1, facecolor='#FFA500', label='. p < 0.10'),
        plt.Rectangle((0,0),1,1, facecolor='#DC143C', label='Not Significant')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()

    # Save to BytesIO for PDF inclusion
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return buf

def create_category_reliability_chart(category_stats):
    """Create a bar chart showing feature category reliability"""
    plt.figure(figsize=(10, 6))

    categories = [cat['Category'] for cat in category_stats]
    reliabilities = [cat['Reliability'] for cat in category_stats]

    bars = plt.bar(categories, reliabilities, color='#4169E1')
    plt.ylabel('Reliability Score (%)', fontsize=10)
    plt.title('Feature Category Reliability Analysis', fontsize=12, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, reliability in zip(bars, reliabilities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{reliability:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return buf

def generate_statistical_report(importance_df, model_metrics, category_stats, output_path=None):
    """
    Generate comprehensive PDF report with statistical significance findings

    Args:
        importance_df: DataFrame with feature importance and statistical significance
        model_metrics: Dict with model performance metrics
        category_stats: List of category reliability statistics
        output_path: Optional output path, defaults to timestamped filename

    Returns:
        str: Path to generated PDF file
    """
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"premier_league_model_report_{timestamp}.pdf"

    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1,  # Center alignment
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=20,
    )

    # Title Page
    story.append(Paragraph("Premier League Prediction Model", title_style))
    story.append(Paragraph("Statistical Significance Report", title_style))
    story.append(Spacer(1, 30))

    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 50))

    # Executive Summary
    story.append(Paragraph("Executive Summary", subtitle_style))

    summary_data = [
        ["Model Performance", ".1f"],
        ["Mean Absolute Error (MAE)", ".3f"],
        ["Accuracy", ".1f"],
        ["", ""],
        ["Statistical Analysis", ""],
        ["Total Features Analyzed", "d"],
        ["Statistically Significant Features", "d"],
        ["Significance Rate", ".1f"],
        ["Most Reliable Category", "s"],
        ["Category Reliability", ".1f"]
    ]

    summary_table_data = [["Metric", "Value"]]
    for metric, format_spec in summary_data:
        if metric == "Model Performance":
            summary_table_data.append([metric, ""])
        elif metric == "Mean Absolute Error (MAE)":
            summary_table_data.append([metric, format_spec])
        elif metric == "Accuracy":
            summary_table_data.append([metric, format_spec])
        elif metric == "":
            summary_table_data.append(["", ""])
        elif metric == "Statistical Analysis":
            summary_table_data.append([metric, ""])
        elif metric == "Total Features Analyzed":
            summary_table_data.append([metric, format_spec])
        elif metric == "Statistically Significant Features":
            summary_table_data.append([metric, format_spec])
        elif metric == "Significance Rate":
            summary_table_data.append([metric, ".1f"])
        elif metric == "Most Reliable Category":
            summary_table_data.append([metric, model_metrics.get('top_category', 'N/A')])
        elif metric == "Category Reliability":
            summary_table_data.append([metric, ".1f"])

    summary_table = Table(summary_table_data, colWidths=[200, 100])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (1, 1), (-1, -1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))

    # Add interpretation
    interpretation = f"""
    This report analyzes {model_metrics['total_features']} features used in the Premier League match outcome prediction model.
    Only {model_metrics['significant_features']} features ({model_metrics['significance_rate']:.1f}%) demonstrate statistical significance at p < 0.05,
    indicating that most features appearing important may be spurious. The {model_metrics['top_category']} category shows the highest reliability
    with {model_metrics['top_reliability']:.1f}% of its features being statistically significant.
    """
    story.append(Paragraph(interpretation, styles['Normal']))
    story.append(Spacer(1, 30))

    # Feature Importance Chart
    story.append(Paragraph("Top Feature Importances", subtitle_style))
    chart_buf = create_feature_importance_chart(importance_df)
    chart_img = Image(chart_buf, width=6*inch, height=4*inch)
    story.append(chart_img)
    story.append(Spacer(1, 20))

    # Top Features Table
    story.append(Paragraph("Detailed Feature Analysis", subtitle_style))
    top_features = importance_df.head(15)[['Feature', 'Importance %', 'Significance', 'P_Value']].copy()
    top_features['Importance %'] = top_features['Importance %'].round(3)
    top_features['P_Value'] = top_features['P_Value'].apply(lambda x: f"{x:.2e}")

    table_data = [['Feature', 'Importance %', 'Significance', 'P-Value']] + top_features.values.tolist()
    features_table = Table(table_data, colWidths=[180, 70, 80, 70])
    features_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (1, 1), (-1, -1), 8),
        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
        ('ALIGN', (3, 1), (3, -1), 'RIGHT'),
    ]))
    story.append(features_table)
    story.append(Spacer(1, 30))

    # Category Reliability Chart
    story.append(Paragraph("Feature Category Reliability", subtitle_style))
    category_chart_buf = create_category_reliability_chart(category_stats)
    category_img = Image(category_chart_buf, width=6*inch, height=4*inch)
    story.append(category_img)
    story.append(Spacer(1, 20))

    # Category Statistics Table
    story.append(Paragraph("Category Reliability Details", styles['Heading3']))
    category_table_data = [['Category', 'Features', 'Significant', 'Reliability %']]
    for cat in category_stats:
        category_table_data.append([
            cat['Category'],
            str(cat['Features']),
            str(cat['Significant']),
            ".1f"
        ])

    category_table = Table(category_table_data, colWidths=[120, 60, 70, 80])
    category_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (1, 1), (-1, -1), 9),
    ]))
    story.append(category_table)
    story.append(Spacer(1, 30))

    # Methodology Section
    story.append(Paragraph("Methodology", subtitle_style))
    methodology_text = """
    <b>Statistical Significance Testing:</b><br/>
    • Features are evaluated using permutation importance with 10 repeats<br/>
    • Z-scores calculated as: (importance - mean) / standard_deviation<br/>
    • P-values derived from two-tailed normal distribution test<br/>
    • Significance levels: *** (p < 0.001), ** (p < 0.01), * (p < 0.05), . (p < 0.10)<br/><br/>

    <b>Model Performance:</b><br/>
    • Multi-class classification (Home/Draw/Away outcomes)<br/>
    • XGBoost classifier with default hyperparameters<br/>
    • 80/20 train/test split with stratification<br/>
    • Evaluation metrics: Mean Absolute Error and Accuracy
    """
    story.append(Paragraph(methodology_text, styles['Normal']))
    story.append(Spacer(1, 20))

    # Footer
    footer_text = f"Report generated by Premier League Prediction System • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=1)
    story.append(Spacer(1, 30))
    story.append(Paragraph(footer_text, footer_style))

    # Build PDF
    doc.build(story)
    return output_path

def generate_quick_report(importance_df, model_metrics, category_stats):
    """
    Generate a quick summary PDF report for download
    Returns PDF as bytes for Streamlit download
    """
    from io import BytesIO

    # Create PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, spaceAfter=30, alignment=1)
    story.append(Paragraph("Premier League Model Report", title_style))
    story.append(Paragraph(f"Generated {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Key Metrics
    story.append(Paragraph("Key Metrics", styles['Heading2']))
    metrics_data = [
        ["MAE", ".3f"],
        ["Accuracy", ".1f"],
        ["Features Analyzed", "d"],
        ["Significant Features", "d"],
        ["Significance Rate", ".1f"]
    ]

    metrics_table = Table([["Metric", "Value"]] + [[metric, format_spec] for metric, format_spec in metrics_data])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))

    # Top 5 Features
    story.append(Paragraph("Top 5 Significant Features", styles['Heading2']))
    top_5 = importance_df.head(5)[['Feature', 'Importance %', 'Significance']].copy()
    top_5['Importance %'] = top_5['Importance %'].round(2)

    top_5_data = [['Feature', 'Importance %', 'Significance']] + top_5.values.tolist()
    top_5_table = Table(top_5_data)
    top_5_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(top_5_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

if __name__ == "__main__":
    # Example usage for testing
    print("PDF Report Generator")
    print("Run from Streamlit app to generate reports")
    print("Example: generate_statistical_report(importance_df, model_metrics, category_stats)")