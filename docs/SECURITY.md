# Security

## Supported use

This is a research prototype. It is not a hosted diagnostic service. There is no production security programme and no bug-bounty programme.

## Reporting

Open a private security advisory on the GitHub repository once the repository exists. Do not file a public issue that contains an image, a credential, or an unpublished vulnerability.

Contact: aksentijeikechukwu@gmail.com

Give the version, the command you ran, and the effect. You should hear back with an acknowledgement even if a fix will take time. This file cannot promise a response time until a maintainer is actually watching the repository.

## Secrets

Never commit `kaggle.json`, API tokens, or cloud credentials. The Colab notebook expects you to upload `kaggle.json` into the session's `~/.kaggle/` path and does not print it. If a token appears in a notebook output, revoke it.

## Medical images

Do not attach identifiable images to issues, pull requests, or a public Gradio share. `--allow-public-share` creates a third-party tunnel. Leave it off unless every image is non-identifying and you accept that tunnel.

The demo stores uploads in temporary process storage. It does not promise zero storage. Shut the process down and remove the temporary directory when you are done.

A readable image file is not proof that the upload is an appropriate or de-identified medical input.

## What this policy does not do

Following this file does not make the model safe for clinical use. There is no clinical authorization to protect.
