; Inno Setup 6 script for UltraEnhance Pro
; Compile with: ISCC installer\UltraEnhancePro.iss   (build.bat does this for you)

#define AppName "UltraEnhance Pro"
#define AppVersion "1.0.0"
#define AppExe "UltraEnhancePro.exe"
#define AppPublisher "UltraEnhance"

[Setup]
AppId={{8C1E6B5A-3F7D-4E2B-9A61-5D0C2F4B7E19}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Install for the current user without admin rights by default.
; The dialog lets the user choose "all users" (Program Files, needs admin).
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir=..\dist
OutputBaseFilename=UltraEnhancePro_Setup
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
WizardStyle=modern
LicenseFile=..\licenses\THIRD_PARTY_NOTICES.txt
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\UltraEnhancePro\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Temporary frames only. Settings, log and the user's enhanced files are kept.
Type: filesandordirs; Name: "{%USERPROFILE}\UltraEnhancePro\temp"
Type: filesandordirs; Name: "{%USERPROFILE}\UltraEnhancePro\jobs"
