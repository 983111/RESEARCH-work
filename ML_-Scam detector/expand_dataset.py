"""
Dataset Generator v2.0 — Realistic, diverse scam vs legitimate messages.

Key fixes vs v1:
  - Messages have REAL variation (not 5 templates + noise)
  - URL features are populated for URL-containing messages
  - No manual_score leakage
  - Balanced classes with realistic linguistic diversity
  - Adversarial edge cases included
"""
import csv
import random
from feature_extractor import extract_features, FEATURE_NAMES

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SCAM MESSAGE TEMPLATES — diverse patterns
# ─────────────────────────────────────────────────────────────────────────────
SCAM_TEMPLATES = [
    # Phishing / account suspension
    "URGENT: Your {bank} account has been suspended. Verify your identity immediately at {shorturl} to avoid permanent closure.",
    "Action Required: We detected suspicious login on your {service} account. Click {shorturl} to confirm it was you.",
    "Your {service} account will be DELETED in 24 hours unless you verify your password now: {shorturl}",
    "Security Alert: Unusual activity detected. Confirm your OTP and PIN at {shorturl} or your account will be locked.",
    "FINAL NOTICE: Your {bank} debit card has been restricted. Update your CVV and billing info here: {shorturl}",

    # Lottery / prize scams
    "Congratulations! You have been selected as a winner of our {prize} prize draw. Claim your reward at {shorturl}",
    "You won ${amount}! This is your final reminder to collect your lottery winnings. Visit {shorturl} to claim.",
    "WINNER ALERT: Your phone number won a ${amount} cash prize. Contact us on WhatsApp to receive payment.",
    "You are our lucky customer! Claim your free {prize} gift card now before it expires: {shorturl}",
    "Our records show you are owed ${amount} in unclaimed funds. Verify your bank account at {shorturl} to receive payment.",

    # Job / money making scams
    "Earn ${amount} per day working from home! No experience needed. DM me on Telegram for details.",
    "We are hiring remote workers. Earn up to ${amount} weekly. Reply with your bank account number to get started.",
    "Guaranteed investment returns of 40% monthly. Limited slots available. Send crypto to get started. WhatsApp now!",
    "Free job offer: Work from home and earn ${amount} daily. Send your SSN and date of birth to apply.",
    "Crypto trading opportunity with guaranteed profit. Invest and double your money in 48 hours. Contact us on Telegram.",

    # Fake tech support
    "WARNING: Your computer is infected with a virus. Call this number immediately to avoid data loss: 1-800-FAKE-HELP",
    "Microsoft Security Alert: Your Windows license has expired. Click {shorturl} to renew and avoid account suspension.",
    "Your device is compromised. Unauthorized access detected. Click {shorturl} and enter your login to secure your account.",
    "Apple ID suspended due to unusual activity. Verify at {badurl} within 12 hours or your account will be permanently deleted.",
    "Your PayPal account is limited. Please confirm your credit card details and social security number at {badurl}",

    # Romance / advance fee
    "Hello dear, I found your contact. I am a wealthy widow with ${amount} I wish to transfer. Please send your bank details.",
    "I am a diplomat and need your help to transfer funds. 30% commission for you. Send your routing number urgently.",
    "You have been chosen to receive a donation of ${amount} from a charitable foundation. Email us your bank account info.",
    "Inheritance claim: You are listed as beneficiary for ${amount}. Contact our lawyer and provide your ID and bank details.",

    # SMS / Delivery scams
    "Your package could not be delivered. Pay the ${amount} customs fee at {shorturl} to release your shipment.",
    "USPS: Your parcel is on hold. Update your delivery address and pay ${amount} fee at {badurl}",
    "DHL: Package awaiting customs clearance. Confirm delivery details and payment at {shorturl} immediately.",
    "Your order has been flagged. Provide your credit card number to verify your identity and release the order.",

    # Fake government
    "IRS NOTICE: You owe ${amount} in back taxes. Pay now at {shorturl} to avoid arrest warrant being issued.",
    "Social Security Administration: Your SSN has been suspended. Call immediately to avoid legal consequences.",
    "You are under investigation for tax fraud. Provide your bank details to cooperate with federal authorities.",
    # Government threats WITHOUT URLs (train model on these)
    "IRS NOTICE: You owe ${amount} in unpaid taxes. Failure to respond will result in immediate arrest.",
    "Final warning: Your Social Security Number has been flagged. Call now or face legal consequences.",
    "Your account is suspended pending fraud investigation. Reply with your PIN to verify identity.",
    # Credential theft without URLs
    "URGENT: Verify your password and OTP immediately or your account will be permanently deleted.",
    "Security check: Reply with your CVV, PIN and account number to confirm your identity.",
    "Your bank needs to verify your social security number and date of birth. Reply now to avoid suspension.",
]

# ─────────────────────────────────────────────────────────────────────────────
# LEGITIMATE MESSAGE TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
LEGIT_TEMPLATES = [
    # Work / professional
    "Hi team, please find the updated project documentation attached. Let me know if you have any questions.",
    "The quarterly report is ready for review. I've shared it on the company drive. Meeting scheduled for Thursday at 2pm.",
    "Reminder: team standup tomorrow at 10am in Conference Room B. Agenda attached.",
    "Please review the pull request I submitted on github.com/org/repo. Happy to discuss the changes.",
    "Here is the link to the tutorial we discussed: youtube.com/watch?v=abc123. Hope it helps!",
    "The client presentation is ready. I've uploaded the slides to the shared folder. Feedback welcome by Friday.",
    "Just a heads up — the server maintenance window is scheduled for Sunday night from 11pm to 2am.",
    "Can everyone please fill out the survey by end of day? The link is in the company Slack channel.",
    "Hi, this is a reminder that your dental appointment is scheduled for Monday at 3:30pm.",
    "Monthly newsletter: Check out our blog post on machine learning trends at our company website.",

    # E-commerce (legitimate)
    "Your order #12345 has been shipped and will arrive by Thursday. Track it at amazon.com/tracking",
    "Your Apple ID was used to sign in on a new device in your location. If this was you, no action is needed.",
    "Your Google account password was changed successfully. If you did not make this change, visit google.com/security",
    "Receipt for your PayPal payment of $49.99 to TechShop Inc. Transaction ID: PP-123456.",
    "Your GitHub repository build succeeded. View the deployment at github.com/your-org/your-repo/actions",

    # Educational / informational
    "Check out this helpful Python tutorial series on youtube.com/pythontutorials",
    "The research paper you requested is available at the university library portal.",
    "Just wanted to share this interesting article on AI safety from the MIT Technology Review.",
    "The meeting minutes from yesterday's session have been posted to the team wiki.",
    "Friendly reminder to complete your annual security training by end of this month.",

    # Personal / casual
    "Hey, are you free for lunch on Wednesday? There's a new place that opened downtown.",
    "Can you send me the recipe you mentioned? My family loved it when you made it last time.",
    "Reminder: book club meeting this Saturday at 6pm. We're discussing the second half of the book.",
    "Thanks for covering my shift last week. I owe you one! Let me know when works for you.",
    "The hiking trail we were looking at has great reviews. Want to plan a trip next month?",

    # Notifications (legitimate)
    "Your two-factor authentication code is 847291. This code expires in 10 minutes. Do not share it.",
    "Password reset requested for your account. If you did not request this, you can ignore this message.",
    "Your subscription renewal is coming up in 7 days. Log in to manage your plan at your account portal.",
    "New comment on your post: Sarah replied to your question in the forum.",
    "Your flight confirmation: AA1234, departing Monday at 8:15am. Check-in opens 24 hours before departure.",
    # Legitimate OTP / security codes (look like scam but are not)
    "Your verification code is 392847. Enter it to complete sign-in. Never share this code with anyone.",
    "Your one-time passcode for account login is 558821. Valid for 5 minutes.",
    "Sign-in attempt from a new device. Your confirmation code is 719234. If this was not you, change your password.",
    # Legitimate server/IT alerts
    "URGENT: Server maintenance tonight at 11 PM. Please save your work and log off before then.",
    "Action required: Your SSL certificate expires in 7 days. Log in to renew it before downtime occurs.",
]

# ─────────────────────────────────────────────────────────────────────────────
# Filler variables
# ─────────────────────────────────────────────────────────────────────────────
BANKS = ['Chase', 'Bank of America', 'Wells Fargo', 'Citibank', 'HSBC', 'TD Bank']
SERVICES = ['Netflix', 'Amazon', 'PayPal', 'Google', 'Apple', 'Microsoft', 'Facebook']
PRIZES = ['$5,000', '$10,000', '$500 gift card', 'luxury vacation', 'iPhone 15', '$2,500 cash']
AMOUNTS = ['500', '1,000', '5,000', '10,000', '250', '2,500', '50,000', '100']
SHORT_URLS = ['bit.ly/verify-now', 'tinyurl.com/secure-login', 'cutt.ly/claimprize',
              'goo.gl/reward2024', 'is.gd/acctverify']
BAD_URLS = ['paypa1.secure-login.tk', 'amazon-verify.ml', 'apple-id.support.pw',
            'google-security.cf', 'login-microsoft.xyz', '192.168.1.1/verify']

SCAM_NOISE = ['', ' Act now!', ' This is urgent!', ' Do not ignore this.',
              ' Reply immediately.', ' Time is running out.', ' ASAP!!!']

LEGIT_NOISE = ['', ' Thanks!', ' Best regards.', ' Let me know if you need anything.',
               ' Hope this helps.', ' Looking forward to hearing from you.']


def fill_template(template):
    return (template
            .replace('{bank}', random.choice(BANKS))
            .replace('{service}', random.choice(SERVICES))
            .replace('{prize}', random.choice(PRIZES))
            .replace('{amount}', random.choice(AMOUNTS))
            .replace('{shorturl}', random.choice(SHORT_URLS))
            .replace('{badurl}', random.choice(BAD_URLS)))


def generate_dataset(n_scam=700, n_legit=700, output_path='scam_dataset.csv'):
    rows = []

    for _ in range(n_scam):
        template = random.choice(SCAM_TEMPLATES)
        text = fill_template(template) + random.choice(SCAM_NOISE)
        features = extract_features(text)
        rows.append(features + [1])

    for _ in range(n_legit):
        template = random.choice(LEGIT_TEMPLATES)
        text = template + random.choice(LEGIT_NOISE)
        features = extract_features(text)
        rows.append(features + [0])

    random.shuffle(rows)

    header = FEATURE_NAMES + ['label']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"✅ Dataset created: {output_path}")
    print(f"   Total samples: {len(rows)} ({n_scam} scam, {n_legit} legit)")
    print(f"   Features: {len(FEATURE_NAMES)}")
    return output_path


if __name__ == '__main__':
    generate_dataset()