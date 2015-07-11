#!/usr/bin/perl

$ok=1;
while(<*.exe>) {
    print "$_  ";
    $result=`./$_`;
    if($result=~/ALL TESTS PASSED/) { print "OK\n"; } else { print "FAILED\n"; $ok=0; }
}
if($ok) { print "ALL TESTS PASSED\n"; }