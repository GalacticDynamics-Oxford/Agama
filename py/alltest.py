#!/usr/bin/python
def alltest():
    import sys,os,glob,subprocess,time
    exe = glob.glob("../exe/test_*.exe")
    py  = [ (sys.executable, f) for f in glob.glob("../py/test_*.py") ]
    allprogs = exe + py
    passed = 0
    failed = 0
    skipped= 0
    unknown= 0
    for prog in allprogs:
        progname = os.path.basename(prog[1] if isinstance(prog, tuple) else prog)
        sys.stdout.write(progname)
        sys.stdout.flush()
        filler = ' ' * (32-len(progname))
        try:
            tstart = time.time()
            proc = subprocess.Popen(prog, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            result = proc.communicate()[0].decode()
            elapsed = filler + "%5.1f s, " % (time.time()-tstart)
            if "ALL TESTS PASSED" in result:
                print(elapsed+"OK")
                passed+=1
            elif "SOME TESTS FAILED" in result or "FAILED TO IMPORT" in result:
                print(elapsed+"FAILED")
                failed+=1
            elif "SKIPPED DUE TO" in result:
                print(elapsed+"SKIPPED")
                skipped+=1
            else:
                print(elapsed+"UNKNOWN")  # not a failure, nor a success
                unknown+=1
        except Exception as e:
            print(filler + "NOT STARTED: "+str(e))
            failed+=1
    print(str(passed)+" TESTS PASSED" + \
        ( ", "+str(failed)+" FAILED" if failed>0 else ", NONE FAILED") +
        ( ", "+str(skipped)+" SKIPPED" if skipped>0 else "") +
        ( ", "+str(unknown)+" UNKNOWN" if unknown>0 else "") )
    return failed==0

if __name__ == '__main__':
    alltest()
