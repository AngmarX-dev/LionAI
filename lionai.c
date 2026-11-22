/*
 * LionAI - lightweight, native file organizer
 * Single-file C11 implementation. No Python, network, or external AI service.
 *
 * Build:
 *   Windows (MSVC): cl /std:c11 /W4 lionai.c
 *   Linux:           cc -std=c11 -O2 -Wall -Wextra -pedantic lionai.c -o lionai
 *
 * Usage:
 *   lionai create <directory> <topic>
 *   lionai organize <directory> <topic> [--apply]
 *
 * Organize is preview-only unless --apply is explicitly supplied.
 * Current MVP classifies by file extension; it does not inspect file contents.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <direct.h>
#define PATH_SEP '\\'
#define MKDIR(path) _mkdir(path)
#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define PATH_SEP '/'
#define MKDIR(path) mkdir(path, 0755)
#endif

#define LIONAI_VERSION "0.1.0"
#define MAX_PATH_LEN 4096
#define MAX_NAME_LEN 256

typedef struct {
    char name[MAX_NAME_LEN];
    char folder[MAX_NAME_LEN];
    const char *extensions;
} FileGroup;

static FileGroup groups[] = {
    {"Image",       "Images",       ".png .jpg .jpeg .gif .bmp .webp .tif .tiff .svg .heic"},
    {"Document",    "Documents",    ".pdf .doc .docx .odt .rtf .txt .md .ppt .pptx .xls .xlsx .csv"},
    {"Audio",       "Audio",        ".mp3 .wav .flac .aac .ogg .m4a .wma"},
    {"Video",       "Videos",       ".mp4 .mkv .mov .avi .webm .wmv .m4v"},
    {"Archive",     "Archives",     ".zip .7z .rar .tar .gz .bz2 .xz"},
    {"Code",        "Code",         ".c .h .cpp .hpp .cs .java .js .ts .py .rs .go .html .css .json .xml .yml .yaml"},
    {"Installer",   "Installers",   ".exe .msi .msix .deb .rpm .apk .appimage"},
    {"Font",        "Fonts",        ".ttf .otf .woff .woff2"},
    {"Database",    "Databases",    ".db .sqlite .sqlite3 .sql"},
    {"Other",       "Other",        ""}
};

#define GROUP_COUNT (sizeof(groups) / sizeof(groups[0]))

static void print_usage(const char *program) {
    printf("LionAI %s - Lightweight file organizer\n\n", LIONAI_VERSION);
    printf("Usage:\n");
    printf("  %s create <directory> <topic>\n", program);
    printf("  %s organize <directory> <topic> [--apply]\n\n", program);
    printf("Examples:\n");
    printf("  %s create \"D:\\\\Work\\\\Photography\" \"Nature photography\"\n", program);
    printf("  %s organize \"D:\\\\Downloads\" \"Personal files\"\n", program);
    printf("  %s organize \"D:\\\\Downloads\" \"Personal files\" --apply\n\n", program);
    printf("Safety: organize only previews changes unless --apply is specified.\n");
}

static int contains_case_insensitive(const char *text, const char *needle) {
    size_t n = strlen(needle);
    if (n == 0) return 1;
    for (; *text; ++text) {
        size_t i = 0;
        while (i < n && text[i] &&
               tolower((unsigned char)text[i]) == tolower((unsigned char)needle[i])) {
            ++i;
        }
        if (i == n) return 1;
    }
    return 0;
}

static int path_join(char *out, size_t cap, const char *a, const char *b) {
    size_t n = strlen(a);
    int written = snprintf(out, cap, "%s%c%s", a,
        (n && (a[n - 1] == '/' || a[n - 1] == '\\')) ? '\0' : PATH_SEP, b);
    if (written < 0 || (size_t)written >= cap) return 0;
    /* Remove the separator placeholder when the input already ends in one. */
    if (n && (a[n - 1] == '/' || a[n - 1] == '\\')) {
        memmove(out + n, out + n + 1, strlen(out + n + 1) + 1);
    }
    return 1;
}

static int is_directory(const char *path) {
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

static int make_directory(const char *path) {
    if (is_directory(path)) return 1;
    if (MKDIR(path) == 0) return 1;
    return errno == EEXIST && is_directory(path);
}

static const char *file_extension(const char *name) {
    const char *dot = strrchr(name, '.');
    if (!dot || dot == name || dot[1] == '\0') return "";
    return dot;
}

static int extension_matches(const char *extension, const char *list) {
    char token[32];
    size_t used = 0;
    while (*list) {
        while (*list && isspace((unsigned char)*list)) ++list;
        used = 0;
        while (*list && !isspace((unsigned char)*list) && used + 1 < sizeof(token)) {
            token[used++] = *list++;
        }
        token[used] = '\0';
        if (used && strlen(extension) == used) {
            size_t i;
            for (i = 0; i < used; ++i) {
                if (tolower((unsigned char)extension[i]) !=
                    tolower((unsigned char)token[i])) break;
            }
            if (i == used) return 1;
        }
        while (*list && !isspace((unsigned char)*list)) ++list;
    }
    return 0;
}

static size_t classify_file(const char *name) {
    const char *ext = file_extension(name);
    size_t i;
    for (i = 0; i + 1 < GROUP_COUNT; ++i) {
        if (extension_matches(ext, groups[i].extensions)) return i;
    }
    return GROUP_COUNT - 1;
}

static int create_topic_structure(const char *root, const char *topic) {
    static const char *general[] = {
        "Documents", "Images", "Audio", "Videos", "Archives", "Other"
    };
    static const char *photo[] = {
        "Images", "Images/Originals", "Images/Edited", "Documents", "Exports"
    };
    static const char *software[] = {
        "Source", "Assets", "Documentation", "Builds", "Tests", "Archives"
    };
    static const char *media[] = {
        "Images", "Audio", "Videos", "Documents", "Exports", "Archives"
    };
    const char **layout = general;
    size_t count = sizeof(general) / sizeof(general[0]);
    size_t i;
    char path[MAX_PATH_LEN];

    if (contains_case_insensitive(topic, "photo") ||
        contains_case_insensitive(topic, "عکس") ||
        contains_case_insensitive(topic, "تصویر")) {
        layout = photo;
        count = sizeof(photo) / sizeof(photo[0]);
    } else if (contains_case_insensitive(topic, "software") ||
               contains_case_insensitive(topic, "game") ||
               contains_case_insensitive(topic, "نرم افزار") ||
               contains_case_insensitive(topic, "بازی")) {
        layout = software;
        count = sizeof(software) / sizeof(software[0]);
    } else if (contains_case_insensitive(topic, "video") ||
               contains_case_insensitive(topic, "media") ||
               contains_case_insensitive(topic, "رسانه") ||
               contains_case_insensitive(topic, "ویدیو")) {
        layout = media;
        count = sizeof(media) / sizeof(media[0]);
    }

    if (!make_directory(root)) {
        fprintf(stderr, "Error: cannot create or access directory: %s\n", root);
        return 0;
    }

    printf("LionAI structure plan for topic: %s\n", topic);
    for (i = 0; i < count; ++i) {
        if (!path_join(path, sizeof(path), root, layout[i])) {
            fprintf(stderr, "Error: path is too long.\n");
            return 0;
        }
        if (!make_directory(path)) {
            fprintf(stderr, "Error: could not create: %s\n", path);
            return 0;
        }
        printf("  [created/exists] %s\n", path);
    }
    printf("Done. No file contents were created or modified.\n");
    return 1;
}

static int name_exists(const char *path) {
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES;
#else
    return access(path, F_OK) == 0;
#endif
}

static int move_file(const char *from, const char *to) {
#ifdef _WIN32
    return MoveFileA(from, to) != 0;
#else
    return rename(from, to) == 0;
#endif
}

static int build_unique_name(char *out, size_t cap, const char *folder,
                             const char *base, const char *extension,
                             unsigned int index) {
    char filename[MAX_NAME_LEN];
    char destination[MAX_PATH_LEN];
    unsigned int attempt = index;
    do {
        snprintf(filename, sizeof(filename), "%s_%03u%s", base, attempt, extension);
        if (!path_join(destination, sizeof(destination), folder, filename)) return 0;
        if (!name_exists(destination)) {
            if (strlen(destination) + 1 > cap) return 0;
            strcpy(out, destination);
            return 1;
        }
        ++attempt;
    } while (attempt < index + 100000U);
    return 0;
}

static int organize_directory(const char *root, const char *topic, int apply) {
    unsigned long planned = 0, moved = 0, skipped = 0;
    unsigned int next_index[GROUP_COUNT] = {0};
#ifdef _WIN32
    char pattern[MAX_PATH_LEN];
    WIN32_FIND_DATAA data;
    HANDLE handle;
    if (!path_join(pattern, sizeof(pattern), root, "*")) {
        fprintf(stderr, "Error: path is too long.\n");
        return 0;
    }
    handle = FindFirstFileA(pattern, &data);
    if (handle == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Error: cannot scan directory: %s\n", root);
        return 0;
    }
    do {
        char source[MAX_PATH_LEN], target_dir[MAX_PATH_LEN], target[MAX_PATH_LEN];
        size_t group;
        const char *ext;
        if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ||
            data.cFileName[0] == '.') continue;
        group = classify_file(data.cFileName);
        ext = file_extension(data.cFileName);
        if (!path_join(source, sizeof(source), root, data.cFileName) ||
            !path_join(target_dir, sizeof(target_dir), root, groups[group].folder) ||
            !build_unique_name(target, sizeof(target), target_dir,
                               groups[group].name, ext, ++next_index[group])) {
            ++skipped;
            continue;
        }
        ++planned;
        printf("  %s -> %s\n", source, target);
        if (apply) {
            if (!make_directory(target_dir) || !move_file(source, target)) {
                fprintf(stderr, "  [failed] %s (error %lu)\n",
                        data.cFileName, (unsigned long)GetLastError());
                ++skipped;
            } else ++moved;
        }
    } while (FindNextFileA(handle, &data));
    FindClose(handle);
#else
    DIR *dir = opendir(root);
    struct dirent *entry;
    if (!dir) {
        fprintf(stderr, "Error: cannot scan directory: %s (%s)\n", root, strerror(errno));
        return 0;
    }
    while ((entry = readdir(dir)) != NULL) {
        char source[MAX_PATH_LEN], target_dir[MAX_PATH_LEN], target[MAX_PATH_LEN];
        struct stat st;
        size_t group;
        const char *ext;
        if (entry->d_name[0] == '.') continue;
        if (!path_join(source, sizeof(source), root, entry->d_name) ||
            lstat(source, &st) != 0 || !S_ISREG(st.st_mode)) continue;
        group = classify_file(entry->d_name);
        ext = file_extension(entry->d_name);
        if (!path_join(target_dir, sizeof(target_dir), root, groups[group].folder) ||
            !build_unique_name(target, sizeof(target), target_dir,
                               groups[group].name, ext, 1)) {
            ++skipped;
            continue;
        }
        ++planned;
        printf("  %s -> %s\n", source, target);
        if (apply) {
            if (!make_directory(target_dir) || !move_file(source, target)) {
                fprintf(stderr, "  [failed] %s (%s)\n", entry->d_name, strerror(errno));
                ++skipped;
            } else ++moved;
        }
    }
    closedir(dir);
#endif
    printf("\nTopic: %s\n", topic);
    printf("Files planned: %lu | moved: %lu | skipped: %lu\n", planned, moved, skipped);
    if (!apply) printf("Preview only. Re-run with --apply to perform these moves.\n");
    else printf("Organization complete. File contents were not rewritten.\n");
    return skipped == 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 ||
        strcmp(argv[1], "-h") == 0) {
        print_usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }
    if (strcmp(argv[1], "--version") == 0) {
        printf("LionAI %s\n", LIONAI_VERSION);
        return 0;
    }
    if (strcmp(argv[1], "create") == 0) {
        if (argc != 4) {
            print_usage(argv[0]);
            return 2;
        }
        return create_topic_structure(argv[2], argv[3]) ? 0 : 1;
    }
    if (strcmp(argv[1], "organize") == 0) {
        int apply = argc == 5 && strcmp(argv[4], "--apply") == 0;
        if (argc != 4 && argc != 5) {
            print_usage(argv[0]);
            return 2;
        }
        if (argc == 5 && !apply) {
            fprintf(stderr, "Unknown option: %s\n", argv[4]);
            return 2;
        }
        if (!is_directory(argv[2])) {
            fprintf(stderr, "Error: target directory does not exist: %s\n", argv[2]);
            return 1;
        }
        return organize_directory(argv[2], argv[3], apply) ? 0 : 1;
    }
    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    print_usage(argv[0]);
    return 2;
}
