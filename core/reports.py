"""Módulo para generación de reportes PDF."""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
        PageBreak, KeepTogether
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab no está disponible, se usará FPDF como fallback")

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    logger.warning("FPDF no está disponible")


class ReportGenerator:
    """Generador de reportes PDF con predicción de riesgo de ACV.
    
    Intenta usar ReportLab primero, si no está disponible usa FPDF como fallback.
    """
    
    def __init__(self):
        """Inicializa el generador de reportes."""
        self.use_reportlab = REPORTLAB_AVAILABLE
        self.use_fpdf = not REPORTLAB_AVAILABLE and FPDF_AVAILABLE
    
    def generate_report(
        self,
        prediction_result: Dict[str, Any],
        input_data: pd.DataFrame,
        output_path: Path,
        recommendations: Optional[list] = None
    ) -> Path:
        """Genera un reporte PDF con los resultados de la predicción.
        
        Args:
            prediction_result: Diccionario con resultados de la predicción.
            input_data: DataFrame con los datos ingresados por el usuario.
            output_path: Ruta donde guardar el reporte PDF.
            recommendations: Lista opcional de recomendaciones personalizadas.
            
        Returns:
            Ruta al archivo PDF generado.
            
        Raises:
            RuntimeError: Si no hay librería de PDF disponible.
        """
        if self.use_reportlab:
            return self._generate_with_reportlab(
                prediction_result, input_data, output_path, recommendations
            )
        elif self.use_fpdf:
            return self._generate_with_fpdf(
                prediction_result, input_data, output_path, recommendations
            )
        else:
            raise RuntimeError(
                "No hay librería de PDF disponible. Instala reportlab o fpdf."
            )
    
    def _generate_with_reportlab(
        self,
        prediction_result: Dict[str, Any],
        input_data: pd.DataFrame,
        output_path: Path,
        recommendations: Optional[List[str]]
    ) -> Path:
        """Genera reporte usando ReportLab."""
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("ReportLab no está disponible")
        
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        # Asegurar que el directorio existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Crear documento
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # Estilos
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Contenido
        story = []
        
        # Título
        story.append(Paragraph("Reporte de Predicción de Riesgo de ACV", title_style))
        story.append(Spacer(1, 0.5*cm))
        
        # Fecha
        fecha = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        story.append(Paragraph(f"<b>Fecha de generación:</b> {fecha}", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
        
        # Resultado de la predicción
        prediction = prediction_result.get('prediction', 'N/A')
        probability = prediction_result.get('probability', 0.0)
        
        # Color según el resultado
        if prediction == "STROKE RISK":
            color = colors.HexColor('#d32f2f')  # Rojo
            risk_level = "ALTO RIESGO"
        else:
            color = colors.HexColor('#388e3c')  # Verde
            risk_level = "BAJO RIESGO"
        
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("RESULTADO DE LA PREDICCIÓN", heading_style))
        
        # Tabla de resultado
        result_data = [
            ['Predicción:', prediction],
            ['Nivel de Riesgo:', risk_level],
            ['Probabilidad:', f"{probability:.1%}"]
        ]
        
        result_table = Table(result_data, colWidths=[4*cm, 10*cm])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (1, 0), (1, 0), color),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ]))
        
        story.append(result_table)
        story.append(Spacer(1, 0.5*cm))
        
        # Datos del paciente
        story.append(Paragraph("DATOS INGRESADOS", heading_style))
        
        # Convertir DataFrame a tabla
        data_dict = input_data.iloc[0].to_dict() if len(input_data) > 0 else {}
        data_rows = [['Campo', 'Valor']]
        
        for key, value in data_dict.items():
            # Formatear valores
            if pd.isna(value):
                value_str = "N/A"
            elif isinstance(value, (int, float)):
                if isinstance(value, float) and value.is_integer():
                    value_str = str(int(value))
                else:
                    value_str = str(value)
            else:
                value_str = str(value)
            
            data_rows.append([str(key).replace('_', ' ').title(), value_str])
        
        data_table = Table(data_rows, colWidths=[6*cm, 8*cm])
        data_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))
        
        story.append(data_table)
        story.append(Spacer(1, 0.5*cm))
        
        # Recomendaciones
        if recommendations:
            story.append(Paragraph("RECOMENDACIONES", heading_style))
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                story.append(Spacer(1, 0.2*cm))
        
        # Nota final
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph(
            "<i>Este reporte es generado automáticamente. "
            "No reemplaza la consulta médica profesional.</i>",
            styles['Normal']
        ))
        
        # Construir PDF
        doc.build(story)
        logger.info(f"Reporte PDF generado exitosamente: {output_path}")
        
        return output_path
    
    def _generate_with_fpdf(
        self,
        prediction_result: Dict[str, Any],
        input_data: pd.DataFrame,
        output_path: Path,
        recommendations: Optional[List[str]]
    ) -> Path:
        """Genera reporte usando FPDF (fallback)."""
        from fpdf import FPDF
        
        # Asegurar que el directorio existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Crear PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Título
        pdf.set_font('Arial', 'B', 16)
        pdf.set_text_color(31, 71, 136)  # Azul oscuro
        pdf.cell(0, 10, 'Reporte de Prediccion de Riesgo de ACV', 0, 1, 'C')
        pdf.ln(5)
        
        # Fecha
        fecha = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, f'Fecha de generacion: {fecha}', 0, 1, 'L')
        pdf.ln(5)
        
        # Resultado
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'RESULTADO DE LA PREDICCION', 0, 1, 'L')
        pdf.ln(3)
        
        prediction = prediction_result.get('prediction', 'N/A')
        probability = prediction_result.get('probability', 0.0)
        
        pdf.set_font('Arial', '', 11)
        pdf.cell(60, 8, 'Prediccion:', 0, 0, 'L')
        pdf.set_font('Arial', 'B', 11)
        if prediction == "STROKE RISK":
            pdf.set_text_color(211, 47, 47)  # Rojo
        else:
            pdf.set_text_color(56, 142, 60)  # Verde
        pdf.cell(0, 8, prediction, 0, 1, 'L')
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 11)
        pdf.cell(60, 8, 'Probabilidad:', 0, 0, 'L')
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, f"{probability:.1%}", 0, 1, 'L')
        pdf.ln(5)
        
        # Datos
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'DATOS INGRESADOS', 0, 1, 'L')
        pdf.ln(3)
        
        data_dict = input_data.iloc[0].to_dict() if len(input_data) > 0 else {}
        pdf.set_font('Arial', '', 10)
        
        for key, value in data_dict.items():
            value_str = str(value) if not pd.isna(value) else "N/A"
            key_str = str(key).replace('_', ' ').title()
            pdf.cell(80, 7, f'{key_str}:', 0, 0, 'L')
            pdf.cell(0, 7, value_str, 0, 1, 'L')
        
        pdf.ln(5)
        
        # Recomendaciones
        if recommendations:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'RECOMENDACIONES', 0, 1, 'L')
            pdf.ln(3)
            
            pdf.set_font('Arial', '', 10)
            for i, rec in enumerate(recommendations, 1):
                # Manejar caracteres especiales
                rec_clean = rec.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 7, f'{i}. {rec_clean}', 0, 'L')
                pdf.ln(2)
        
        # Nota
        pdf.ln(5)
        pdf.set_font('Arial', 'I', 9)
        pdf.multi_cell(0, 7, 
            'Este reporte es generado automaticamente. '
            'No reemplaza la consulta medica profesional.', 0, 'L')
        
        # Guardar
        pdf.output(str(output_path))
        logger.info(f"Reporte PDF generado exitosamente (FPDF): {output_path}")
        
        return output_path
