from preprocessing_functions import clean_text, cleaning_text_lenghth,add_comment_length,encoding_date

import googleapiclient.discovery
import pandas as pd

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyDr-UnL8Nst1yHOE85cRxsHAvHk25jaKk0"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)


def getcomments(video):
  request = youtube.commentThreads().list(
      part="snippet",
      videoId=video,
      maxResults=100
  )

  comments = []

  # Execute the request.
  response = request.execute()

  # Get the comments from the response.
  for item in response['items']:
      comment = item['snippet']['topLevelComment']['snippet']
      public = item['snippet']['isPublic']
      comments.append([
          comment['textOriginal'],
          comment['likeCount'],
          comment['publishedAt']
      ])

  while (1 == 1):
    try:
     nextPageToken = response['nextPageToken']
    except KeyError:
     break
    nextPageToken = response['nextPageToken']
    # Create a new request object with the next page token.
    nextRequest = youtube.commentThreads().list(part="snippet", videoId=video, maxResults=100, pageToken=nextPageToken)
    # Execute the next request.
    response = nextRequest.execute()
    # Get the comments from the next response.
    for item in response['items']:
      comment = item['snippet']['topLevelComment']['snippet']
      public = item['snippet']['isPublic']
      comments.append([
         comment['textOriginal'],
          comment['likeCount'],
          comment['publishedAt']
      ])

  df2 = pd.DataFrame(comments, columns=['CommentText','Likes','PublishedAt'])
  
  

  #preprocessing the data
  
  df2=add_comment_length(df2)
  df2=cleaning_text_lenghth(df2)
  df2["CommentText"] = df2["CommentText"].apply(clean_text)
  df2["CommentText"] = df2["CommentText"].fillna("").astype(str)
  df2=encoding_date(df2)



  return df2

df= getcomments("T609xCMZtGQ")
df.head()