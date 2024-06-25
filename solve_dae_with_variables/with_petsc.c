#include <petsc.h>

typedef struct {
    Vec p;
    Vec u;
    Mat Ju;
    PetscReal t0;
    PetscReal tf;
    PetscReal dt;
} DAEContext;

PetscErrorCode CreateParametersVector(DAEContext *ctx) {
    PetscErrorCode ierr;
    ierr = VecCreateSeq(PETSC_COMM_WORLD, 2, &ctx->p); CHKERRQ(ierr);
    ierr = VecSetValue(ctx->p, 0, 1.0, INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecSetValue(ctx->p, 1, 2.0, INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(ctx->p); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(ctx->p); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode CreateStateVector(DAEContext *ctx) {
    PetscErrorCode ierr;
    ierr = VecCreateSeq(PETSC_COMM_WORLD, 3, &ctx->u); CHKERRQ(ierr);
    ierr = VecSetValue(ctx->u, 0, 2.0, INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecSetValue(ctx->u, 1, 2.0, INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecSetValue(ctx->u, 2, -3.0, INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(ctx->u); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(ctx->u); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode EvalIFunction(TS ts, PetscReal t, Vec u, Vec udot, Vec f, void *ctx) {
    DAEContext *dae = (DAEContext*)ctx;
    PetscErrorCode ierr;
    const PetscScalar *u_array, *udot_array;
    PetscScalar *f_array, *p_array;

    ierr = VecGetArrayRead(u, &u_array); CHKERRQ(ierr);
    ierr = VecGetArrayRead(udot, &udot_array); CHKERRQ(ierr);
    ierr = VecGetArray(f, &f_array); CHKERRQ(ierr);
    ierr = VecGetArray(dae->p, &p_array); CHKERRQ(ierr);

    f_array[0] = udot_array[0] + p_array[0] * u_array[0] - p_array[1] * u_array[1] * (1.0 - u_array[0] - u_array[1]);
    f_array[1] = udot_array[1] - p_array[1] * u_array[0] * u_array[0] + u_array[1];
    f_array[2] = u_array[2] - 1.0 + u_array[0] + u_array[1];

    ierr = VecRestoreArrayRead(u, &u_array); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(udot, &udot_array); CHKERRQ(ierr);
    ierr = VecRestoreArray(f, &f_array); CHKERRQ(ierr);
    ierr = VecRestoreArray(dae->p, &p_array); CHKERRQ(ierr);

    ierr = VecAssemblyBegin(f); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(f); CHKERRQ(ierr);

    return 0;
}

PetscErrorCode EvalIJacobian(TS ts, PetscReal t, Vec u, Vec udot, PetscReal shift, Mat Ju, Mat P, void *ctx) {
    DAEContext *dae = (DAEContext*)ctx;
    PetscErrorCode ierr;
    const PetscScalar *u_array, *p_array;
    PetscInt row[3] = {0, 1, 2}, col[3] = {0, 1, 2};
    PetscScalar v[9];

    ierr = VecGetArrayRead(u, &u_array); CHKERRQ(ierr);
    ierr = VecGetArrayRead(dae->p, &p_array); CHKERRQ(ierr);

    v[0] = shift + p_array[0] + p_array[1] * u_array[1];
    v[1] = -p_array[1] * u_array[0] + p_array[1] * 2.0 * u_array[1];
    v[2] = 0.0;
    v[3] = -p_array[1] * 2.0 * u_array[0];
    v[4] = shift + 1.0;
    v[5] = 0.0;
    v[6] = 1.0;
    v[7] = 1.0;
    v[8] = 1.0;

    ierr = MatSetValues(Ju, 3, row, 3, col, v, INSERT_VALUES); CHKERRQ(ierr);

    ierr = VecRestoreArrayRead(u, &u_array); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(dae->p, &p_array); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(Ju, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Ju, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    if (P != Ju) {
        ierr = MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

    return 0;
}

int main(int argc, char **argv) {
    PetscErrorCode ierr;
    DAEContext dae;
    TS ts;

    ierr = PetscInitialize(&argc, &argv, NULL, NULL); if (ierr) return ierr;

    dae.t0 = 0.0;
    dae.tf = 1.0;
    dae.dt = 0.001;

    ierr = CreateParametersVector(&dae); CHKERRQ(ierr);
    ierr = CreateStateVector(&dae); CHKERRQ(ierr);

    ierr = MatCreateDense(PETSC_COMM_WORLD, 3, 3, 3, 3, NULL, &dae.Ju); CHKERRQ(ierr);
    ierr = MatSetUp(dae.Ju); CHKERRQ(ierr);

    ierr = TSCreate(PETSC_COMM_WORLD, &ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts, TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetType(ts, TSCN); CHKERRQ(ierr);
    ierr = TSSetIFunction(ts, NULL, EvalIFunction, &dae); CHKERRQ(ierr);
    ierr = TSSetIJacobian(ts, dae.Ju, dae.Ju, EvalIJacobian, &dae); CHKERRQ(ierr);

    ierr = TSSetTime(ts, dae.t0); CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts, dae.tf); CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, dae.dt); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts); CHKERRQ(ierr);

    ierr = TSSolve(ts, dae.u); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "Forward solve complete.\n"); CHKERRQ(ierr);
    ierr = VecView(dae.u, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    ierr = VecDestroy(&dae.p); CHKERRQ(ierr);
    ierr = VecDestroy(&dae.u); CHKERRQ(ierr);
    ierr = MatDestroy(&dae.Ju); CHKERRQ(ierr);
    ierr = TSDestroy(&ts); CHKERRQ(ierr);

    ierr = PetscFinalize();
    return ierr;
}
