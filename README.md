# Hybrid-DNS-Sinkhole-AI
Online Gambling DNS Query Traffic Detection System Using Machine Learning and Pi-hole

PROJECT BACKGROUND
Online gambling in Indonesia has escalated into a nationwide crisis, with 12.3 million active players and a cumulative financial turnover of IDR 1,163.9 trillion. Syndicates exhibit high infrastructure resilience by utilizing Domain Generation Algorithms (DGA) to rapidly generate thousands of unregistered domains daily, while simultaneously exploiting server vulnerabilities to inject malicious redirections into high-reputation authoritative websites. Conventional enforcement relies heavily on reactive, static blacklist-based filtering, which creates a systemic blind spot as it cannot detect zero-day DGA domains or hijacked subdomains. This project develops a proactive, cost-effective detection model that shifts machine learning capabilities directly to a local single-board computer, ensuring all DNS filtering and decision-making occurs within the user's network without compromising privacy.

TECHNOLOGY & ARCHIETECTURE
The system employs a Hybrid DNS Sinkhole dual-path engine architecture running on a Raspberry Pi 4 Model B (4 GB RAM) and a MikroTik hAP Lite router.
1. Traffic Interception: A MikroTik router utilizes Destination NAT rules to transparently intercept all client queries targeting port 53 and forcibly redirect them to the local proxy.
2. Machine Learning Middleware (First Line of Defense): An inline Python proxy processes unseen domains by extracting 12 lexical features and applying Term Frequency-Inverse Document Frequency (TF-IDF) weighting on 3-5 character N-grams. A Random Forest model, trained on a corpus of 5.1 million domains, evaluates the extracted feature vectors.
3. Cost-Sensitive Thresholding: The decision boundary is calibrated to a specific probability threshold of 0.3559 via F3-Score optimization, intentionally minimizing false negatives to enforce strict blocking rules against camouflage techniques.
4. Static Filtering (Second Line of Defense): Queries classified as safe (score below 0.3559) are forwarded to a backend Pi-hole interface on port 5353 for conventional ad, tracker, and malware blocking.
5. Fast-Path Caching: Processed domains are cached in RAM, allowing recurring DNS queries to be resolved instantly with an NXDOMAIN response (0.0.0.0) without executing re-inference.

EVALUATION & RESULTS
The system was validated through both offline corpus evaluation and online testing against active real-world attack vectors, including SEO poisoning and web defacements.
1. Classification Accuracy: Online testing against 600 active domains yielded a blocking rate of 98.00% and an allowing rate of 95.33%. The false negative rate was contained at 2.00% and the false positive rate at 4.67%, successfully meeting the specification targets of maintaining a minimum 95% accuracy and a maximum 5% error tolerance.
2. Quality of Service (QoS): Under daily usage conditions from seven client devices, the system achieved a QoS index of 3.33 (Good) based on TIPHON standards. The integration of the inline machine learning inference layer maintained an average latency of 46.67 ms, perfectly matching the baseline network performance prior to the sinkhole's implementation.
3. Scalability Limits: During stress testing under extreme load, the QoS index dropped to 2.67 (Fair) due to computational bottlenecks in the edge device's ARM processor. The heavy volume of feature extraction caused socket buffer queues to build up, resulting in latency surging to 416.39 ms and a query loss rate of 23.22%, despite reaching a peak throughput of 48.69 Queries Per Second (QPS)

*Dataset Information
Due to GitHub's file size limitations, the full dataset containing 5.1 million rows of DNS traffic could not be uploaded directly. However, the full dataset can be accessed and downloaded via the following link:
https://drive.google.com/drive/folders/13ZEV6ksSrWvNLf7R-T7OG11Tlcx8LAcY?usp=sharing
