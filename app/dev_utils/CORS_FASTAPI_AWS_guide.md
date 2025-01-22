Plan:
1. use a load balanced environment
2. make sure the app redirects to https -- I already have an ssl certificate:
   - SSL: identifier: ca368661-cb87-41fd-8cb1-ec0f0744dec7 arn: arn:aws:acm:us-east-1:307946648330:certificate/ca368661-cb87-41fd-8cb1-ec0f0744dec7
3. I have a Dockerfile so no need for Procfile
