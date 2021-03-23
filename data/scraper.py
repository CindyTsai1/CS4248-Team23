filename = 'raw_fb_posts.csv'
headers = ['post_id', 'text', 'post_text', 'shared_text', 'time', 'image', 'video', 
         'video_thumbnail', 'video_id', 'num_reactions', 'num_comments', 'num_shares', 'post_url', 
         'link', 'user_id', 'username', 'is_live', 'factcheck', 'shared_post_id', 
         'shared_time', 'shared_user_id', 'shared_username', 'shared_post_url', 
         'available', 'images', 'w3_fb_url', 'fetched_time',
         'num_like', 'num_love', 'num_support', 'num_haha','num_wow', 'num_sorry', 'num_angry']
reaction_types = ['like', 'love', 'support', 'haha', 'wow', 'sorry', 'angry'] # support == care, sorry == sad

import csv
import datetime
import time
from facebook_scraper import get_posts, _scraper

def scrape():
    posts = []
    count = 0
    cutoff_date = datetime.datetime(2018, 1, 1)

    print('Starting at: ' + str(datetime.datetime.now()))
    start = time.time()

    with open(filename, mode='w') as file:
        csv_writer = csv.DictWriter(file, fieldnames=headers)
        csv_writer.writeheader()

        # extra_info=True to attempt to get detailed reactions
        # pages=None to iterate till end of all posts
        for post in get_posts('nuswhispers', pages=None, extra_info=True, timeout=30):
            if post['time'] < cutoff_date:
                print("Reached cutoff date, terminating.")
                break

            count += 1

            # rename
            post['num_reactions'] = post.pop('likes')
            post['num_comments'] = post.pop('comments')
            post['num_shares'] = post.pop('shares')

            # flatten reactions dict, if there is one
            has_reactions = 'no'
            if 'reactions' in post:
                reactions = post['reactions']
                for key in reactions:
                    post[f'num_{key}'] = reactions[key]
                post.pop('reactions')
                has_reactions = 'yes'

            print(f'#{count} | {post["time"]} | post_id: {post.get("post_id")} | text: {post.get("text") is not None} | reactions: {has_reactions}')

            csv_writer.writerow(post)
            file.flush()

            posts.append(post)
    end = time.time()
    print('Time taken: ' + str(end - start))
    
if __name__ == '__main__':
    scrape()