# -*- coding: utf-8 -*-
def main():
    import query_video, sys, tempfile, os
    with tempfile.TemporaryDirectory() as results:
        CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')
        query_video.query_video(sys.argv[2], sys.argv[1], CONFIG_FILE, results, 1)
