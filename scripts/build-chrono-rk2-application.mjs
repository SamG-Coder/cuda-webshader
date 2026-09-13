// Reuse the configured upstream demo flags, including Eigen's alignment ABI.
import {readFileSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {execFileSync} from 'node:child_process';
if(process.platform!=='win32')throw Error('This capture build uses the configured Windows Chrono build.');
const revision=execFileSync('git',['-C','.local/chrono-upstream','rev-parse','HEAD'],{encoding:'utf8'}).trim();
if(revision!=='a92c6f72f422fbcafe0b37125d4070cb6a3b5803')throw Error('Chrono capture requires the pinned upstream revision.');
const commands=JSON.parse(readFileSync('.local/chrono-build/compile_commands.json'));
const demo=commands.find(x=>x.file.endsWith('demo_FSI-SPH_DamBreak.cpp'));
if(!demo||!demo.command.includes(' -c '))throw Error('Configure the original Chrono dam-break target first.');
const command=demo.command.replace(/\/Fo(?:"[^"]+"|\S+)/,'/Fo.local/chrono-rk2-application.obj').replace(/\/Fd(?:"[^"]+"|\S+)/,'/Fd.local/chrono-rk2-application.pdb').replace(/ -c .+$/,' -c "'+resolve('tests/chrono-rk2-application.cpp')+'"');
const cudaPath=process.env.CUDA_PATH;if(!cudaPath)throw Error('CUDA_PATH is required for the native marker capture.');
const batch='.local/chrono-rk2-application-build.cmd';
writeFileSync(batch,'@echo off\r\ncall "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"\r\n'+command+'\r\nif errorlevel 1 exit /b 1\r\ncl /nologo .local/chrono-rk2-application.obj /Fe.local/chrono-build/bin/chrono-rk2-application.exe /link /LIBPATH:.local/chrono-build/lib Chrono_core.lib Chrono_fsi.lib Chrono_fsisph.lib yaml-cpp.lib /LIBPATH:"'+cudaPath+'\\lib\\x64" cudart.lib\r\n');
execFileSync('cmd.exe',['/d','/c',batch.replaceAll('/','\\')],{stdio:'inherit'});
