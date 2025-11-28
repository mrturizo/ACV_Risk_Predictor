"""
Script para compilar la aplicación de escritorio a .exe usando PyInstaller.

Uso:
    python build_exe.py
"""

import subprocess
import sys
import os
from pathlib import Path

def build_exe():
    """Compila la aplicación a ejecutable."""
    project_root = Path(__file__).parent.parent
    app_desktop_dir = Path(__file__).parent
    
    print("=" * 60)
    print("Compilando ACV Risk Predictor a ejecutable")
    print("=" * 60)
    print()
    
    # Verificar que PyInstaller esté instalado
    try:
        import PyInstaller
        print(f"✓ PyInstaller encontrado: {PyInstaller.__version__}")
    except ImportError:
        print("❌ PyInstaller no está instalado.")
        print("   Instalando PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller instalado")
    
    print()
    print("Iniciando compilación...")
    print()
    
    # Cambiar al directorio de app_desktop
    import os
    original_dir = Path.cwd()
    os.chdir(app_desktop_dir)
    
    try:
        # Ejecutar PyInstaller
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            "ACV_Risk_Predictor.spec"
        ]
        
        print(f"Ejecutando: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, check=True)
        
        print()
        print("=" * 60)
        print("✓ Compilación completada exitosamente!")
        print("=" * 60)
        print()
        print(f"Ejecutable generado en: {app_desktop_dir / 'dist' / 'ACV_Risk_Predictor.exe'}")
        print()
        print("Próximos pasos:")
        print("1. Probar el ejecutable")
        print("2. Crear instalador con InnoSetup usando installer_script.iss")
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("❌ Error durante la compilación")
        print("=" * 60)
        print(f"Error: {e}")
        return False
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ Error inesperado")
        print("=" * 60)
        print(f"Error: {e}")
        return False
    finally:
        os.chdir(original_dir)
    
    return True


if __name__ == "__main__":
    import os
    build_exe()

