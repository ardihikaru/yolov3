import simplejson as json
import time
import cv2

# default = dumped kabeh.
def redis_set(redis_client, key, value, expired=None):
    value = json.dumps(value)

    if expired is not None:
        option = [value, expired]
        redis_client.set(key, *option)
    else:
        redis_client.set(key, value)

def redis_get(redis_client, key):
    data = None
    try:
        data = json.loads(redis_client.get(key))
    except:
        pass
    finally:
        return data

def redis_get_all_keys(redis_client):
    data = None
    try:
        data = json.loads(redis_client.keys())
    except:
        pass
    finally:
        return data

def pub(my_redis, channel, message):
    my_redis.publish(channel, message)

def sub(my_redis, channel, func, consumer_name=None):
    pubsub = my_redis.pubsub()
    pubsub.subscribe([channel])
    for item in pubsub.listen():
        func(consumer_name, item['data'])

def frame_producer(my_redis, frame_id, ret, frame, save_path, channel):
    if ret:
        # Save image
        # print("###### save_path = ", save_path)
        t0 = time.time()
        # save_path = "/home/ardi/devel/nctu/5g-dive/docker-yolov3/output_frames/hasil.jpg"
        cv2.imwrite(save_path, frame)
        print(".. image is saved in (%.3fs)" % (time.time() - t0))

        # Publish information
        t0 = time.time()
        data = {
            "frame_id": frame_id,
            "img_path": save_path
        }
        p_mdata = json.dumps(data)
        print(" .. Start publishing")
        # my_redis.publish('stream', p_mdata)
        my_redis.publish(channel, p_mdata)
        print(".. frame is published in (%.3fs)" % (time.time() - t0))