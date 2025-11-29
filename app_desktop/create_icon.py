"""
Script para generar iconos profesionales para ACV Risk Predictor.

Genera icon.ico (con múltiples tamaños) e icon.png para la aplicación desktop.
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys

# Colores del tema médico
COLOR_PRIMARY = '#1f77b4'  # Azul médico
COLOR_WHITE = '#ffffff'
COLOR_ACCENT = '#ff7f0e'  # Naranja de alerta
COLOR_DARK = '#262730'  # Gris oscuro

# Tamaños para el icono ICO (Windows requiere múltiples tamaños)
ICO_SIZES = [16, 32, 48, 64, 128, 256]
PNG_SIZE = 256  # Tamaño para el PNG de referencia


def create_icon_image(size: int) -> Image.Image:
    """
    Crea una imagen de icono con diseño médico.
    
    Args:
        size: Tamaño de la imagen en píxeles.
        
    Returns:
        Imagen PIL con el diseño del icono.
    """
    # Crear imagen con fondo transparente
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Calcular dimensiones relativas
    padding = size // 8
    center_x = size // 2
    center_y = size // 2
    radius = (size - padding * 2) // 2
    
    # Dibujar círculo de fondo (azul médico)
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        fill=COLOR_PRIMARY,
        outline=COLOR_WHITE,
        width=max(1, size // 32)
    )
    
    # Dibujar símbolo de cerebro simplificado
    # Usar formas geométricas para representar un cerebro
    brain_scale = radius * 0.7
    
    # Lado izquierdo del cerebro (forma de media luna)
    brain_left = [
        (center_x - brain_scale * 0.3, center_y - brain_scale * 0.4),
        (center_x - brain_scale * 0.8, center_y - brain_scale * 0.2),
        (center_x - brain_scale * 0.6, center_y),
        (center_x - brain_scale * 0.8, center_y + brain_scale * 0.2),
        (center_x - brain_scale * 0.3, center_y + brain_scale * 0.4),
    ]
    draw.polygon(brain_left, fill=COLOR_WHITE)
    
    # Lado derecho del cerebro
    brain_right = [
        (center_x + brain_scale * 0.3, center_y - brain_scale * 0.4),
        (center_x + brain_scale * 0.8, center_y - brain_scale * 0.2),
        (center_x + brain_scale * 0.6, center_y),
        (center_x + brain_scale * 0.8, center_y + brain_scale * 0.2),
        (center_x + brain_scale * 0.3, center_y + brain_scale * 0.4),
    ]
    draw.polygon(brain_right, fill=COLOR_WHITE)
    
    # Dibujar símbolo de alerta/riesgo (triángulo con exclamación) en la esquina superior derecha
    alert_size = size // 4
    alert_x = size - alert_size - padding
    alert_y = padding
    
    # Triángulo de alerta
    alert_triangle = [
        (alert_x + alert_size // 2, alert_y),
        (alert_x, alert_y + alert_size),
        (alert_x + alert_size, alert_y + alert_size),
    ]
    draw.polygon(alert_triangle, fill=COLOR_ACCENT, outline=COLOR_WHITE, width=max(1, size // 64))
    
    # Exclamación en el triángulo (solo para tamaños grandes)
    if size >= 32:
        exclamation_width = max(1, size // 32)
        exclamation_height = alert_size // 2
        exclamation_x = alert_x + alert_size // 2
        exclamation_y = alert_y + alert_size // 3
        
        # Línea vertical
        draw.rectangle(
            [exclamation_x - exclamation_width // 2, exclamation_y,
             exclamation_x + exclamation_width // 2, exclamation_y + exclamation_height],
            fill=COLOR_WHITE
        )
        
        # Punto (solo para tamaños >= 64)
        if size >= 64:
            dot_size = max(2, size // 32)
            draw.ellipse(
                [exclamation_x - dot_size // 2, alert_y + alert_size - dot_size - padding // 2,
                 exclamation_x + dot_size // 2, alert_y + alert_size - padding // 2],
                fill=COLOR_WHITE
            )
    
    return img


def create_ico_file(output_path: Path):
    """
    Crea un archivo ICO con múltiples tamaños.
    
    Args:
        output_path: Ruta donde guardar el archivo .ico
    """
    images = []
    for size in ICO_SIZES:
        img = create_icon_image(size)
        images.append(img)
    
    # Guardar como ICO (PIL soporta guardar múltiples tamaños en un ICO)
    images[0].save(
        str(output_path),
        format='ICO',
        sizes=[(img.width, img.height) for img in images]
    )
    print(f"✓ Icono ICO creado: {output_path}")


def create_png_file(output_path: Path):
    """
    Crea un archivo PNG de referencia.
    
    Args:
        output_path: Ruta donde guardar el archivo .png
    """
    img = create_icon_image(PNG_SIZE)
    img.save(str(output_path), format='PNG')
    print(f"✓ Icono PNG creado: {output_path}")


def main():
    """Función principal para generar los iconos."""
    # Obtener el directorio donde está este script
    script_dir = Path(__file__).parent
    
    # Rutas de salida
    ico_path = script_dir / 'icon.ico'
    png_path = script_dir / 'icon.png'
    
    print("Generando iconos para ACV Risk Predictor...")
    print(f"Directorio: {script_dir}")
    print()
    
    try:
        # Crear iconos
        create_ico_file(ico_path)
        create_png_file(png_path)
        
        print()
        print("✓ Iconos generados exitosamente!")
        print(f"  - {ico_path}")
        print(f"  - {png_path}")
        print()
        print("Los iconos están listos para usar en:")
        print("  - PyInstaller (.spec)")
        print("  - InnoSetup (installer_script.iss)")
        print("  - Ventana de Tkinter (main_tkinter.py)")
        
    except ImportError as e:
        print("ERROR: No se pudo importar Pillow (PIL)")
        print("Instala Pillow con: pip install Pillow")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR al generar iconos: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

