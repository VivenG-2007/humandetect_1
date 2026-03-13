import sys
try:
    import mediapipe as mp
    print("mediapipe imported")
    print("hasattr(mp, 'solutions'):", hasattr(mp, 'solutions'))
    try:
        from mediapipe.solutions import pose as mp_pose
        print("from mediapipe.solutions import pose worked")
    except Exception as e:
        print("from mediapipe.solutions import pose failed:", e)
    
    try:
        import mediapipe.solutions.pose as mp_pose2
        print("import mediapipe.solutions.pose worked")
    except Exception as e:
        print("import mediapipe.solutions.pose failed:", e)
        
except Exception as e:
    print("mediapipe import failed:", e)
