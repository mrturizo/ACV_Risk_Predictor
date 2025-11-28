; Script de InnoSetup para crear instalador .exe
; ACV Risk Predictor - Instalador Profesional

[Setup]
; Información básica de la aplicación
AppName=ACV Risk Predictor
AppVersion=1.0.0
AppPublisher=ACV Risk Predictor Team
AppPublisherURL=https://github.com/mrturizo/ACV_Risk_Predictor
AppSupportURL=https://github.com/mrturizo/ACV_Risk_Predictor/issues
AppUpdatesURL=https://github.com/mrturizo/ACV_Risk_Predictor/releases
AppCopyright=Copyright (C) 2024 ACV Risk Predictor Contributors
DefaultDirName={pf}\ACV_Risk_Predictor
DefaultGroupName=ACV Risk Predictor
AllowNoIcons=yes
LicenseFile=LICENSE.txt
InfoBeforeFile=README_INSTALLER.txt
OutputDir=dist
OutputBaseFilename=ACV_Risk_Predictor_Setup_v1.0.0
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern

; Icono del instalador (usar el mismo que la app si está disponible)
SetupIconFile=icon.ico

; Información de versión
VersionInfoVersion=1.0.0
VersionInfoCompany=ACV Risk Predictor Team
VersionInfoDescription=Aplicación para predicción de riesgo de ACV usando Machine Learning
VersionInfoCopyright=Copyright (C) 2024
VersionInfoProductName=ACV Risk Predictor
VersionInfoProductVersion=1.0.0

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; Archivo ejecutable principal
Source: "dist\ACV_Risk_Predictor.exe"; DestDir: "{app}"; Flags: ignoreversion signonce

; Incluir modelos si existen (descomentar cuando tengas modelos .pkl)
; Source: "..\models\*.pkl"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs

; Incluir config.py si existe (opcional)
; Source: "..\config.py"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Icono en el menú de inicio
Name: "{group}\ACV Risk Predictor"; Filename: "{app}\ACV_Risk_Predictor.exe"; IconFilename: "{app}\ACV_Risk_Predictor.exe"
Name: "{group}\{cm:UninstallProgram,ACV Risk Predictor}"; Filename: "{uninstallexe}"

; Icono en el escritorio (opcional, según tarea)
Name: "{autodesktop}\ACV Risk Predictor"; Filename: "{app}\ACV_Risk_Predictor.exe"; Tasks: desktopicon; IconFilename: "{app}\ACV_Risk_Predictor.exe"

; Icono en la barra de inicio rápido (opcional, según tarea)
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\ACV Risk Predictor"; Filename: "{app}\ACV_Risk_Predictor.exe"; Tasks: quicklaunchicon; IconFilename: "{app}\ACV_Risk_Predictor.exe"

[Run]
; Opción para ejecutar la aplicación después de la instalación
Filename: "{app}\ACV_Risk_Predictor.exe"; Description: "{cm:LaunchProgram,ACV Risk Predictor}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Limpiar archivos temporales al desinstalar (opcional)
Type: filesandordirs; Name: "{app}\data\outputs\*"
Type: filesandordirs; Name: "{app}\data\uploads\*"

[Code]
// Código personalizado para verificar requisitos (opcional)
function InitializeSetup(): Boolean;
begin
  Result := True;
  // Aquí puedes agregar verificaciones de requisitos del sistema
end;
