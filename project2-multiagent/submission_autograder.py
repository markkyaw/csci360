#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64


import bz2, base64
exec(bz2.decompress(base64.b64decode('QlpoOTFBWSZTWVgJb/QAAB/fgGQQSve3Uq8X3g+//9+6QAK8ZAAA5hMAmAEwmE0wAACZNNAxCaT1J6TynkgGg0ANAAAAADmEwCYATCYTTAAAJk00DCRQQTCnqbQnolPym1TwUGnpPUeU9PVP1NT1HmpJJCdanzwIPT94KEHGI22KDcODP4PX/DMgESEi4a6TE+1H0IL2j/Vimeszyq+vPKxSZB2HMSUsLua72K7JLjVWNk4NtEBrULZo6SfZw56Pgbowa3aYmbgBrSEANcJB8UU8TZQxCZVjIYokEEAFUORUMlEwJ8UxvpQ89vhUDGzHjQd8bLBkFi1uISBAQESEoclwWiFo3mU5yhdBRErCg4LSfWNrt/taFfybZ4GEnu67z59lujG+O/uYgy34m1eZ5dy8iawRHjwiIl6HodxuIV7DvqHWehZFUTSzz4neXhNcRlJecCiJ36MShcbSw16LImKY9qv2tw4LxVxWsGU7atNlJM91ij9FM2nI0JWUCMgs6zV7eJauW5eJWqzpZSefyiVDWnFYekihNG33FkJhRB/8iZPWkcZIn9GpJBoyYXc4PTR85kYZ6VKKpRNjY22gaIqR8C5qJ8kZMPeKJLNbW94+TWrVbH46DLT0VakNU7U0eAifWEVArYMpNg3jBY0KqWWt6NtQr00GY6jYaQySCHNSUBSTZonuoVVKbG1vJ/J/DqyeuThM7ytK3MHSERsbTCMLoyFK5qYaF050NGkXJpjS1o3rUshRQa4MpzdrXgTMu00+SsSkucLSld1ptTDQUdS2WmIFgfmUKla4FygEOxGwM2amsixYb2LPEqVaXa77vW68MHKaxKRxpNpUsHWsiOaRGAfj1PThqWSVN8TmHGBas0hEIlgT87rJLXOCK1qUkZBTrCDsVyz/vHCO5II4LmvRfbKVsb1QdqxPuf+LuSKcKEgsBLf6AA==')))
