@echo off
rem ============================================================================
rem  run_pipeline.bat - build the Docker images and run the LiDAR pipeline
rem
rem      3DFin -> split -> SmartQSM / arbor -> compare
rem
rem  Starts the hydra-zen bridge (bridge\README.md), which launches every stage
rem  image itself.  Run  run_pipeline.bat --help  for the options.
rem
rem  Exit code: 0 ok, 1 error, 2 bad command line; after the run it is the exit
rem  code of the bridge.
rem
rem  Notes for maintainers
rem   * Keep CRLF line endings (see .gitattributes) and plain ASCII in this file.
rem   * Part 1 (command line and paths) runs with delayed expansion OFF, so that
rem     an exclamation mark in a path or an override survives.  Part 2 turns it
rem     ON: every value is read there as !VAR!, which is safe for special
rem     characters such as ampersand, caret, percent and parentheses.
rem   * cmd splits unquoted arguments at comma, semicolon and equals sign, so
rem     the text after "--" is cut out of the raw argument list and passed on
rem     verbatim.
rem   * --gpu builds the GPU image in addition to the others (compose profile
rem     gpu) and appends smartqsm.gpu=true plus, unless the extras after "--"
rem     already name a smartqsm.config, the GPU preset.  The extras stay in
rem     front of the generated overrides on purpose (hydra flags such as -m
rem     must come first).
rem   * Not supported: a double quote inside an argument, or a caret used in
rem     the console to escape an ampersand (put the argument in "quotes").
rem ============================================================================
setlocal EnableExtensions DisableDelayedExpansion

rem ---- 1. Where am I? -------------------------------------------------------
set "START_DIR=%CD%"
if "%START_DIR:~-1%"=="\" set "START_DIR=%START_DIR:~0,-1%"
cd /d "%~dp0"
if errorlevel 1 set "ERRMSG=cannot change to the script folder: %~dp0" & goto :fail
set "PROJECT_DIR=%CD%"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
rem Same folder with forward slashes: what compose.yaml needs for the bind mounts.
set "HOST_PROJECT_DIR=%PROJECT_DIR:\=/%"

rem ---- 2. Command line ------------------------------------------------------
set "INPUT="
set "REF="
set "RUN_NAME="
set "RUN_NAME_X="
set "INPUT_X="
set "STAGES="
set "STAGES_GIVEN="
set "NO_BUILD="
set "BUILD_ONLY="
set "GPU="
set "EXTRA="
set "CONFIG_X="
set "ERRMSG="
set "STAGE_LIST="
set "INPUT_FULL="
set "INPUT_STEM="
set "REF_FULL="

:next_arg
if "%~1"=="" goto :args_done
if /i "%~1"=="-h" goto :usage_ok
if /i "%~1"=="--help" goto :usage_ok
if "%~1"=="/?" goto :usage_ok
if "%~1"=="--" goto :opt_extra
if /i "%~1"=="--ref" goto :opt_ref
if /i "%~1"=="--run" goto :opt_run
if /i "%~1"=="--stages" goto :opt_stages
if /i "%~1"=="--no-build" set "NO_BUILD=1" & shift & goto :next_arg
if /i "%~1"=="--build-only" set "BUILD_ONLY=1" & shift & goto :next_arg
if /i "%~1"=="--gpu" set "GPU=1" & shift & goto :next_arg
set "ARG=%~1"
if "%ARG:~0,1%"=="-" set "ERRMSG=unknown option: %~1" & goto :usage_fail
if defined INPUT set "ERRMSG=unexpected argument: %~1  (quote paths that contain spaces, commas or semicolons; hydra overrides go after --)" & goto :usage_fail
set "INPUT=%~1"
shift
goto :next_arg

:opt_ref
if "%~2"=="" set "ERRMSG=--ref needs a file name" & goto :usage_fail
set "REF=%~2"
shift
shift
goto :next_arg

:opt_run
if "%~2"=="" set "ERRMSG=--run needs a name" & goto :usage_fail
set "RUN_NAME=%~2"
shift
shift
goto :next_arg

:opt_stages
rem A list such as split,compare arrives as several words (cmd splits at commas),
rem so take every word up to the next option.
set "STAGES_GIVEN=1"
shift
:stages_more
if "%~1"=="" goto :next_arg
set "ARG=%~1"
if "%ARG:~0,1%"=="-" goto :next_arg
set "STAGES=%STAGES% %~1"
shift
goto :stages_more

:opt_extra
rem Everything after the first " -- " is handed to hydra exactly as typed.
set EXTRA_RAW= %*
shift
if "%~1"=="" goto :args_done
set EXTRA=%EXTRA_RAW:* -- =%
rem Remember run_name=... and input=... (cmd shows them as two words) for later.
:extra_scan
if "%~1"=="" goto :args_done
if "%~1"=="run_name" set "RUN_NAME_X=%~2"
if "%~1"=="input" set "INPUT_X=1"
if "%~1"=="smartqsm.config" set "CONFIG_X=1"
set "ARG=%~1"
if "%ARG:~0,16%"=="smartqsm.config=" set "CONFIG_X=1"
shift
goto :extra_scan

:args_done
if defined BUILD_ONLY if defined NO_BUILD set "ERRMSG=--build-only and --no-build contradict each other" & goto :usage_fail
if defined BUILD_ONLY goto :part2

rem ---- 2a. Validate (no docker needed yet) ----------------------------------
if defined STAGES_GIVEN if not defined STAGES set "ERRMSG=--stages needs a comma separated list, e.g. smartqsm,compare" & goto :usage_fail
set "STAGE_BAD="
set "HAS_DFIN="
if not defined STAGES set "HAS_DFIN=1" & goto :stages_done
echo "%STAGES%"| findstr /r /x /c:"\"[A-Za-z0-9_, ;]*\"" >nul
if errorlevel 1 set "ERRMSG=--stages: use stage names separated by commas, e.g. dfin,split" & goto :usage_fail
for %%S in (%STAGES%) do call :add_stage %%S
if defined STAGE_BAD goto :usage_fail
if not defined STAGE_LIST set "ERRMSG=--stages: no stage name given" & goto :usage_fail
:stages_done

if not defined RUN_NAME goto :run_ok
echo "%RUN_NAME%"| findstr /r /x /c:"\"[A-Za-z0-9_][A-Za-z0-9._-]*\"" >nul
if errorlevel 1 set "ERRMSG=--run: use letters, digits, dot, underscore and hyphen only, starting with a letter or digit (it becomes the folder data\out\NAME)" & goto :usage_fail
:run_ok

if defined INPUT goto :input_ok
if defined INPUT_X goto :input_ok
if defined HAS_DFIN set "ERRMSG=no input file given, but stage dfin needs one. To run only later stages on an earlier run use --stages ... --run NAME" & goto :usage_fail
if not defined RUN_NAME if not defined RUN_NAME_X set "ERRMSG=without an input file --run NAME is required" & goto :usage_fail
:input_ok

rem ---- 2b. Make the file arguments absolute ---------------------------------
if not defined INPUT goto :input_path_done
set "P=%INPUT%"
call :abs_path || goto :fail
set "INPUT_FULL=%P%"
for %%F in ("%P%") do set "INPUT_STEM=%%~nF"
:input_path_done
if not defined REF goto :ref_path_done
set "P=%REF%"
call :abs_path || goto :fail
set "REF_FULL=%P%"
:ref_path_done

rem ---- Part 2: delayed expansion ON -----------------------------------------
:part2
setlocal EnableDelayedExpansion

rem ---- 3. Files must live inside the project (the containers only see it) ---
set "REL_INPUT="
set "REL_REF="
set "PREFIX=!PROJECT_DIR!\"
call :prefix_len
if not defined INPUT_FULL goto :rel_input_done
set "P=!INPUT_FULL!"
set "WHAT=input file"
call :to_rel || goto :fail
set "REL_INPUT=!P!"
:rel_input_done
if not defined REF_FULL goto :rel_ref_done
set "P=!REF_FULL!"
set "WHAT=reference file"
call :to_rel || goto :fail
set "REL_REF=!P!"
:rel_ref_done

rem ---- 4. Docker ------------------------------------------------------------
where docker >nul 2>&1
if errorlevel 1 set "ERRMSG=docker was not found on PATH. Install Docker Desktop for Windows, start it, and open a new terminal." & goto :fail
docker info >nul 2>&1
if errorlevel 1 set "ERRMSG=the Docker daemon does not answer. Start Docker Desktop, wait until it says Engine running, then run this again." & goto :fail

echo(
echo === LiDAR pipeline ===================================================
echo project folder : !PROJECT_DIR!
echo HOST_PROJECT_DIR: !HOST_PROJECT_DIR!
if defined REL_INPUT echo input          : !REL_INPUT!
if defined REL_REF echo reference      : !REL_REF!

rem ---- 5. Build the images (BuildKit cache makes repeats fast) --------------
rem Four images; with --gpu also lidar-smartqsm-gpu (compose profile gpu, several GB on the first build).
set "BUILD_CMD=docker compose build"
if defined GPU set "BUILD_CMD=docker compose --profile gpu build"
if defined NO_BUILD goto :build_skipped

call :check_submodules
if not defined MISSING goto :submodules_ok
echo [setup] missing !MISSING! - running: git submodule update --init --recursive
where git >nul 2>&1
if errorlevel 1 set "ERRMSG=git was not found on PATH, but the submodule !MISSING! is missing. Install Git for Windows, or clone the project with --recurse-submodules." & goto :fail
git submodule update --init --recursive
if errorlevel 1 set "ERRMSG=git submodule update failed (see the output above)." & goto :fail
call :check_submodules
if defined MISSING set "ERRMSG=!MISSING! is still missing after git submodule update." & goto :fail
:submodules_ok

if defined GPU echo [build] GPU image requested: lidar-smartqsm-gpu is built in addition (the first build downloads several GB)
echo [build] !BUILD_CMD!
!BUILD_CMD!
if errorlevel 1 set "ERRMSG=docker compose build failed (see the output above). Fix the error and run again, or use --no-build to skip building." & goto :fail
echo [build] done
if defined BUILD_ONLY goto :build_only_done
goto :build_finished

:build_skipped
echo [build] skipped (--no-build)
:build_finished

rem ---- 6. Run the bridge ----------------------------------------------------
set "OVR="
if defined REL_INPUT set OVR=!OVR! "input='!REL_INPUT!'"
if defined REL_REF set OVR=!OVR! "reference='!REL_REF!'"
if defined RUN_NAME set OVR=!OVR! "run_name=!RUN_NAME!"
if defined STAGE_LIST set OVR=!OVR! "stages=[!STAGE_LIST!]"
if defined GPU set OVR=!OVR! "smartqsm.gpu=true"
if defined GPU if not defined CONFIG_X set OVR=!OVR! "smartqsm.config=spconv-contraction-LEAFOFF-GPU"

echo(
rem Hydra flags (-m, --cfg ...) must come before the key=value overrides, so the user's extras go first.
echo [run] docker compose run --rm -T bridge python -m lidar_bridge !EXTRA!!OVR!
docker compose run --rm -T bridge python -m lidar_bridge !EXTRA!!OVR!
set "RC=!ERRORLEVEL!"

set "OUT_NAME=!RUN_NAME!"
if defined RUN_NAME_X set "OUT_NAME=!RUN_NAME_X!"
if not defined OUT_NAME set "OUT_NAME=!INPUT_STEM!"
echo(
if not "!RC!"=="0" goto :run_failed
echo Done. Output folder: !PROJECT_DIR!\data\out\!OUT_NAME!
exit /b 0

:run_failed
>&2 echo The pipeline FAILED with exit code !RC!.
>&2 echo Log and partial results: !PROJECT_DIR!\data\out\!OUT_NAME!
exit /b !RC!

:build_only_done
echo(
if defined GPU (echo Done. All images are built, including lidar-smartqsm-gpu. Next: run_pipeline.bat INPUT.laz --ref REFERENCE.csv --gpu) else echo Done. All images are built. Next: run_pipeline.bat INPUT.laz --ref REFERENCE.csv
exit /b 0

rem ===========================================================================
rem  Helpers.  Only reached with call or goto; the main flow always exits above.
rem ===========================================================================

:usage_ok
call :usage
exit /b 0

:usage_fail
call :print_error
call :usage >&2
exit /b 2

:fail
call :print_error
exit /b 1

:print_error
setlocal EnableDelayedExpansion
>&2 echo(
>&2 echo ERROR: !ERRMSG!
endlocal
exit /b 0

:usage
echo run_pipeline.bat - build the Docker images and run the LiDAR pipeline
echo     3DFin - split - SmartQSM / arbor - compare   (hydra-zen bridge)
echo(
echo Usage:
echo   run_pipeline.bat INPUT.laz [options] [-- HYDRA_OVERRIDES...]
echo   run_pipeline.bat --build-only [--gpu]
echo   run_pipeline.bat --stages LIST --run NAME [options]
echo(
echo INPUT.laz and --ref files must be inside this project folder, e.g. in data\in.
echo Paths may be absolute or relative; quote paths that contain spaces.
echo(
echo Options:
echo   --ref FILE        reference tree table (.csv or .xlsx) for the compare stage
echo   --run NAME        run name = output folder data\out\NAME (default: input file name)
echo   --stages LIST     comma separated subset of dfin,split,smartqsm,arbor,compare
echo                     (default: all; they always run in that fixed order). Stages
echo                     after dfin need no INPUT.laz when --run names an earlier run.
echo   --no-build        do not build the images first (they must already exist)
echo   --build-only      build the images and stop
echo   --gpu             also build the GPU image lidar-smartqsm-gpu (CUDA, several GB, once)
echo                     and run SmartQSM on the GPU: smartqsm.gpu=true and, unless you give
echo                     smartqsm.config after --, smartqsm.config=spconv-contraction-LEAFOFF-GPU.
echo                     Needs an NVIDIA GPU, its driver and Docker Desktop GPU support (WSL2).
echo   -h, --help        show this help
echo   --                everything after it goes to hydra unchanged, for example:
echo                       dry_run=true  dfin.normalize=true  split.max_axis_dist=2.5
echo                       smartqsm.config=space-colonization
echo(
echo Examples:
echo   run_pipeline.bat Example_data\plot_01.laz --ref data\in\ref.csv
echo   run_pipeline.bat data\in\plot_01.laz --run r1 --no-build -- dry_run=true
echo   run_pipeline.bat --stages smartqsm,compare --run r1 --ref data\in\ref.csv
echo   run_pipeline.bat --build-only --gpu
echo   run_pipeline.bat --stages smartqsm,compare --run r1 --ref data\in\ref.csv --no-build --gpu
echo(
echo Exit code: 0 ok, 1 error, 2 bad command line, otherwise the bridge's exit code.
exit /b 0

:add_stage
rem Part 1 helper: check one stage name, append it to STAGE_LIST.
if defined STAGE_BAD exit /b 1
set "STAGE_OK="
for %%V in (dfin split smartqsm arbor compare) do if "%~1"=="%%V" set "STAGE_OK=1"
if not defined STAGE_OK set "STAGE_BAD=1" & set "ERRMSG=unknown stage: %~1  (valid: dfin, split, smartqsm, arbor, compare; give INPUT.laz before --stages)" & exit /b 1
if "%~1"=="dfin" set "HAS_DFIN=1"
if not defined STAGE_LIST set "STAGE_LIST=%~1" & exit /b 0
set "STAGE_LIST=%STAGE_LIST%,%~1"
exit /b 0

:abs_path
rem Part 1 helper.  In: P = path as typed.  Out: P = absolute path.
rem A relative path is looked up in the folder the script was started from
rem first, then in the project folder.
echo "%P%"| findstr /c:"*" /c:"?" >nul
if not errorlevel 1 set "ERRMSG=wildcards are not allowed in file names: %P%" & exit /b 1
set "P=%P:/=\%"
if "%P:~1,1%"==":" goto :abs_normalize
if "%P:~0,1%"=="\" goto :abs_normalize
if exist "%START_DIR%\%P%" set "P=%START_DIR%\%P%"
:abs_normalize
for %%F in ("%P%") do set "P=%%~fF"
exit /b 0

:prefix_len
rem Part 2 helper.  PREFIX_LEN = number of characters in PREFIX.
set /a PREFIX_LEN=0
:prefix_len_loop
if "!PREFIX:~%PREFIX_LEN%,1!"=="" exit /b 0
set /a PREFIX_LEN+=1
goto :prefix_len_loop

:to_rel
rem Part 2 helper.  In: P = absolute path, WHAT = what the file is for.
rem Out: P = path relative to the project, with forward slashes (for hydra).
if not exist "!P!" set "ERRMSG=!WHAT! not found: !P!" & exit /b 1
if exist "!P!\" set "ERRMSG=!WHAT! is a folder, a file is needed: !P!" & exit /b 1
set "HEAD=!P:~0,%PREFIX_LEN%!"
if /i not "!HEAD!"=="!PREFIX!" set "ERRMSG=!WHAT! is outside the project folder !PROJECT_DIR! - copy it into !PROJECT_DIR!\data\in and use that copy: !P!" & exit /b 1
set "P=!P:~%PREFIX_LEN%!"
set "P=!P:\=/!"
if not "!P:'=!"=="!P!" set "ERRMSG=!WHAT! has a single quote in its path, which hydra cannot take - rename it: !P!" & exit /b 1
exit /b 0

:check_submodules
rem Part 2 helper.  MISSING = first marker file of a git submodule that is absent (empty if all are there).
set "MISSING="
for %%F in (third_party\3DFin\pyproject.toml third_party\SmartQSM\requirements.txt third_party\arbor\DESCRIPTION) do if not exist "%%F" set "MISSING=%%F"
exit /b 0
