import cv2

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)

    # Only look inside this box, so the face is ignored.
    x1, y1, x2, y2 = 80, 80, 380, 420
    roi = frame[y1:y2, x1:x2]

    # Keep skin-colored pixels. Put the hand on a plain, non-skin background.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 40, 60), (25, 180, 255))
    mask = cv2.medianBlur(mask, 7)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fingers = 0

    if contours:
        hand = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand) > 5000:
            cv2.drawContours(frame, [hand], -1, (0, 255, 0), 2, offset=(x1, y1))
            hull = cv2.convexHull(hand, returnPoints=False)
            if hull is not None and len(hull) > 3 and len(hand) > 3:
                defects = cv2.convexityDefects(hand, hull)
                gaps = 0
                if defects is not None:
                    # A deep dip in the outline is the gap between two fingers.
                    # reshape covers both (n, 4) and (n, 1, 4) layouts from OpenCV.
                    for defect in defects.reshape(-1, 4):
                        if defect[3] > 8000:
                            gaps += 1
                fingers = min(gaps + 1, 5)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Fingers: {fingers}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Finger count", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
