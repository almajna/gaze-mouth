#import <Foundation/Foundation.h>
#include <Python.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        NSBundle *bundle = NSBundle.mainBundle;
        NSString *runtime = [bundle.resourcePath stringByAppendingPathComponent:@"runtime"];
        NSString *envFile = [[NSUserDefaults standardUserDefaults] stringForKey:@"CheckoutEnvironmentFile"];
        if (envFile.length) {
            setenv("CLICKY_ENV_FILE", envFile.fileSystemRepresentation, 0);
        }

        PyPreConfig preconfig;
        PyPreConfig_InitIsolatedConfig(&preconfig);
        preconfig.utf8_mode = 1;
        PyStatus status = Py_PreInitialize(&preconfig);
        if (PyStatus_Exception(status))
            Py_ExitStatusException(status);

        PyConfig config;
        PyConfig_InitIsolatedConfig(&config);
        config.write_bytecode = 0;
        config.parse_argv = 0;
        status = PyConfig_SetBytesString(&config, &config.home, runtime.fileSystemRepresentation);
        if (!PyStatus_Exception(status))
            status = PyConfig_SetBytesString(&config, &config.executable, bundle.executablePath.fileSystemRepresentation);
        if (!PyStatus_Exception(status))
            status = PyConfig_SetString(&config, &config.run_module, L"clicky");
        if (!PyStatus_Exception(status))
            status = PyConfig_SetBytesArgv(&config, argc, argv);
        if (!PyStatus_Exception(status))
            status = Py_InitializeFromConfig(&config);
        PyConfig_Clear(&config);
        if (PyStatus_Exception(status))
            Py_ExitStatusException(status);
        return Py_RunMain();
    }
}
