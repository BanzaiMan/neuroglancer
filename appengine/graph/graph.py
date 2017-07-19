from lru import LRU

#TODO compute capacity based on average graph size
# and instance memory use env variable GAE_MEMORY_MB
capacity = 5

def evicted(key, value):
  print "removing: %s" % (key)
chunks = LRU(5, callaback=evicted)

def download_chunk(chunk_key):
    pass

def get_chunk(chunk_key):
    if chunk_key not in chunks:
        chunks[chunk_key] = download_chunk(chunk_key)
    return chunks[chunk_key]



class ChunkedGraph(object):
    pass


class NodeHandler(webapp2.RequestHandler):
    pass

class MergeHandler(webapp2.RequestHandler):
    pass

class SplitHandler(webapp2.RequestHandler):
    pass

class SubgraphHandler(webapp2.RequestHandler):
    pass

class ChildrenHandler(webapp2.RequestHandler):
    pass


app = webapp2.WSGIApplication([
    (r'/v1/node/(\d+)/?', NodeHandler),
    (r'/v1/merge/(\d+),(\d+)/?', MergeHandler),
    (r'/v1/split/(\d+),(\d+)/?', SplitHandler),
    (r'/v1/subgraph/?', SubgraphHandler),
    (r'/v1/children/(\d+)/?', ChildrenHandler),
], debug=True)
