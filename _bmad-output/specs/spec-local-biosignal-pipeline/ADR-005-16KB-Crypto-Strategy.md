# ADR-005: Android 16 KB Page Size Compatibility - Crypto Strategy

## Status
Accepted (28 July 2026)

## Context
Starting November 1st, 2025, Google Play requires all new apps and updates targeting Android 15+ to support 16 KB page sizes.
The application currently uses `net.zetetic:android-database-sqlcipher:4.5.4` for SQLite database encryption.
During conformance auditing, `libsqlcipher.so` was found to be misaligned (compiled with legacy 4 KB ELF LOAD segment alignment).
Because SQLCipher is a third-party precompiled library that fails the `zipalign -c -P 16 -v 4` check, the application will crash on Android 15+ devices with 16 KB page sizes.

## Decision
We will remove `net.zetetic:android-database-sqlcipher` entirely and revert to the standard Android Room implementation. 
For data-at-rest security, we will rely on Android's native Full-Disk Encryption (FDE) / File-Based Encryption (FBE) and hardware-backed Android Keystore.
If specific fields (e.g., auth tokens) need encryption, they will be stored in `EncryptedSharedPreferences`. 

## Consequences
- **Positive**: Resolves the 16 KB alignment issue, preventing crashes on Android 15+ devices.
- **Positive**: Reduces APK size and runtime memory overhead by removing the SQLCipher native library.
- **Negative**: The SQLite database files on disk are no longer encrypted at the application level. However, they remain protected by the OS sandbox and FBE, which is standard for most clinical and enterprise applications unless rooted.
